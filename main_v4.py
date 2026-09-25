import os
import re
import gc
import time
import random
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from PIL import Image

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score

from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna

# =========================
# Config
# =========================

DATA_DIR = "./data/petfinder/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images/images/train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "images/images/test")

TEXT_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "efficientnet_b0"

IMAGE_SIZE = 224
BATCH_SIZE_IMG = 64

TFIDF_MAX_FEATURES = 500
COLOR_BINS = 32  # per channel → 32*3 = 96

PCA_IMG_DIM = 192
PCA_TXT_DIM = 96

N_OPTUNA_TRIALS = 20
RANDOM_SEED = 42

SUBMISSION_PATH = os.path.join(DATA_DIR, "submission.csv")

# =========================
# Utils
# =========================

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def quadratic_weighted_kappa(y_true, y_pred):
    """
    y_true, y_pred: integer labels 0..N or 1..N
    We'll map to 0..3 internally.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def optimize_thresholds(y_true, y_pred_cont):
    """
    Simple 3-threshold optimization for 4 classes (0..3 or 1..4).
    We assume target labels are 1..4 in PetFinder.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred_cont = np.array(y_pred_cont, dtype=float)

    # Start from percentiles
    qs = np.percentile(y_pred_cont, [25, 50, 75])
    best_t = qs.copy()
    best_kappa = -1.0

    # Small random search around initial thresholds
    for _ in range(200):
        t1 = best_t[0] + np.random.uniform(-0.3, 0.3)
        t2 = best_t[1] + np.random.uniform(-0.3, 0.3)
        t3 = best_t[2] + np.random.uniform(-0.3, 0.3)
        if not (t1 < t2 < t3):
            continue

        y_pred_cls = np.digitize(y_pred_cont, [t1, t2, t3]) + 1  # classes 1..4
        kappa = quadratic_weighted_kappa(y_true, y_pred_cls)
        if kappa > best_kappa:
            best_kappa = kappa
            best_t = np.array([t1, t2, t3])

    return best_t, best_kappa

# =========================
# Load data
# =========================

print("Train dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test dir exists:", os.path.exists(TEST_IMG_DIR))

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("\nПеревірка 1: Загальна інформація")
print("TRAIN:", train.shape)
print("TEST :", test.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nПеревірка 2: Типи колонок")
print(train.dtypes)

print("\nПеревірка 3: Пропуски")
na = train.isnull().sum()
print(na[na > 0].sort_values(ascending=False))

print("\nПеревірка 4: Цільова змінна")
print(train["AdoptionSpeed"].value_counts().sort_index())

train["Description"] = train["Description"].fillna("").apply(clean_text)
test["Description"] = test["Description"].fillna("").apply(clean_text)

print("\nПеревірка 7: Довжина текстів")
desc_len = train["Description"].str.len()
print(desc_len.describe())

# =========================
# Photo stats
# =========================

print("\nПеревірка 8: Кількість фотографій")
train_counts = Counter()
for f in os.listdir(TRAIN_IMG_DIR):
    if f.endswith(".jpg"):
        pet_id = f.split("-")[0]
        train_counts[pet_id] += 1

vals = list(train_counts.values())
print("Min:", min(vals))
print("Mean:", np.mean(vals))
print("Max:", max(vals))
print("Median:", np.median(vals))

print("\nПеревірка 9: Розподіл фото")
photo_hist = Counter(vals)
for k, v in sorted(photo_hist.items()):
    print(k, "photos ->", v)

# =========================
# SentenceTransformer embeddings
# =========================

print("\nЗавантаження SentenceTransformer (MiniLM)...")
model_text = SentenceTransformer(TEXT_EMB_MODEL_NAME)

train_text_emb = model_text.encode(
    train["Description"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
test_text_emb = model_text.encode(
    test["Description"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nПеревірка 10.1: Перевірка embedding")
print(train_text_emb.shape)
print(test_text_emb.shape)

# =========================
# Image embeddings
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

image_model = timm.create_model(
    IMAGE_MODEL_NAME,
    pretrained=True,
    num_classes=0
)
image_model.eval()
image_model.to(device)

transform_img = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

class PetImageDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir
        self.pet_ids = df["PetID"].tolist()

    def __len__(self):
        return len(self.pet_ids)

    def __getitem__(self, idx):
        pet_id = self.pet_ids[idx]
        # choose first photo if multiple
        # filenames: {PetID}-{PhotoNumber}.jpg
        # we just take "-1.jpg" if exists, else any
        candidates = []
        for f in os.listdir(self.img_dir):
            if f.startswith(pet_id) and f.endswith(".jpg"):
                candidates.append(f)
        if len(candidates) == 0:
            # dummy black image
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        else:
            # take first
            img_path = os.path.join(self.img_dir, sorted(candidates)[0])
            img = Image.open(img_path).convert("RGB")
        img = transform_img(img)
        return img

def compute_image_embeddings(df, img_dir):
    dataset = PetImageDataset(df, img_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_IMG, shuffle=False)

    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = image_model(batch)
            emb = emb.cpu().numpy()
            all_emb.append(emb)
    all_emb = np.concatenate(all_emb, axis=0)
    return all_emb

print("\nОбчислення image embeddings...")
print("Train image index size:", len(train))
print("Test image index size:", len(test))

train_img_emb = compute_image_embeddings(train, TRAIN_IMG_DIR)
test_img_emb = compute_image_embeddings(test, TEST_IMG_DIR)

print("Train image emb shape:", train_img_emb.shape)
print("Test image emb shape:", test_img_emb.shape)

# =========================
# TF-IDF
# =========================

print("\nОбчислення TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2
)
tfidf_train = tfidf.fit_transform(train["Description"].tolist()).toarray()
tfidf_test = tfidf.transform(test["Description"].tolist()).toarray()

print("TF-IDF shapes:", tfidf_train.shape, tfidf_test.shape)

# =========================
# Color histograms
# =========================

def compute_color_histograms(df, img_dir, bins=COLOR_BINS):
    hist_list = []
    for pet_id in df["PetID"].tolist():
        candidates = []
        for f in os.listdir(img_dir):
            if f.startswith(pet_id) and f.endswith(".jpg"):
                candidates.append(f)
        if len(candidates) == 0:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        else:
            img_path = os.path.join(img_dir, sorted(candidates)[0])
            img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        # per-channel hist
        hists = []
        for c in range(3):
            channel = img_np[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
            hists.append(hist)
        hists = np.concatenate(hists)
        hist_list.append(hists)
    return np.array(hist_list)

print("\nОбчислення color histograms...")
train_color = compute_color_histograms(train, TRAIN_IMG_DIR, bins=COLOR_BINS)
test_color = compute_color_histograms(test, TEST_IMG_DIR, bins=COLOR_BINS)

print("Train color hist:", train_color.shape[0])
print("Test color hist:", test_color.shape[0])
print("Color hist shapes:", train_color.shape, test_color.shape)

# =========================
# Photo count feature
# =========================

photo_count_map = train_counts  # Counter from earlier

def get_photo_counts(df, img_dir, counter_map):
    counts = []
    for pet_id in df["PetID"].tolist():
        c = counter_map.get(pet_id, 0)
        if c == 0:
            # count files directly
            c = 0
            for f in os.listdir(img_dir):
                if f.startswith(pet_id) and f.endswith(".jpg"):
                    c += 1
        counts.append(c)
    return np.array(counts).reshape(-1, 1)

train_photo_counts = get_photo_counts(train, TRAIN_IMG_DIR, photo_count_map)
test_photo_counts = get_photo_counts(test, TEST_IMG_DIR, photo_count_map)

print("\nTrain photo count stats:")
print(train_photo_counts.min(), train_photo_counts.mean(), train_photo_counts.max())

# =========================
# PCA for image/text embeddings
# =========================

print("\nPCA для image/text...")
pca_img = PCA(n_components=PCA_IMG_DIM, random_state=RANDOM_SEED)
pca_txt = PCA(n_components=PCA_TXT_DIM, random_state=RANDOM_SEED)

train_img_pca = pca_img.fit_transform(train_img_emb)
test_img_pca = pca_img.transform(test_img_emb)

train_txt_pca = pca_txt.fit_transform(train_text_emb)
test_txt_pca = pca_txt.transform(test_text_emb)

print("PCA image:", train_img_pca.shape, test_img_pca.shape)
print("PCA text :", train_txt_pca.shape, test_txt_pca.shape)

# =========================
# Final feature matrix
# =========================

X_train = np.concatenate(
    [train_img_pca, train_txt_pca, tfidf_train, train_color, train_photo_counts],
    axis=1
)
X_test = np.concatenate(
    [test_img_pca, test_txt_pca, tfidf_test, test_color, test_photo_counts],
    axis=1
)

y_train = train["AdoptionSpeed"].values.astype(int)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)

# Map labels to 1..4 (already so), but keep as is.

# =========================
# Train/val split
# =========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_train
)

print("Train split:", X_tr.shape, y_tr.shape)
print("Val split  :", X_val.shape, y_val.shape)

# =========================
# Optuna + CatBoost (CPU)
# =========================

def objective(trial):
    depth = trial.suggest_int("depth", 4, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
    iterations = trial.suggest_int("iterations", 500, 1200)

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        task_type="CPU"
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    val_pred_cont = model.predict(X_val)
    thresholds, kappa = optimize_thresholds(y_val, val_pred_cont)
    return kappa

print("\nRunning Optuna for CatBoost (це може зайняти час)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_params
best_val_qwk = study.best_value

print("Best CatBoost params:", best_params)
print("Best CatBoost val QWK:", best_val_qwk)

# =========================
# Train final CatBoost on full train
# =========================

final_cat = CatBoostRegressor(
    depth=best_params["depth"],
    learning_rate=best_params["learning_rate"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    iterations=best_params["iterations"],
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=RANDOM_SEED,
    verbose=False,
    task_type="CPU"
)

final_cat.fit(X_train, y_train, verbose=False)

# thresholds from val again (for CatBoost-only)
val_pred_cat = final_cat.predict(X_val)
cat_thresholds, cat_val_qwk = optimize_thresholds(y_val, val_pred_cat)

print("CatBoost thresholds:", cat_thresholds)
print("CatBoost val QWK  :", cat_val_qwk)

# =========================
# LightGBM (LGBMRegressor) on full train
# =========================

print("\nTraining LightGBM (LGBMRegressor)...")

model_lgb = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=RANDOM_SEED,
    n_estimators=1500
)

model_lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    early_stopping_rounds=200,
    verbose=False
)

# Retrain on full train with best n_estimators
best_n_estimators = model_lgb.best_iteration_ if hasattr(model_lgb, "best_iteration_") else 1500

final_lgb = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=RANDOM_SEED,
    n_estimators=best_n_estimators
)

final_lgb.fit(X_train, y_train, verbose=False)

# =========================
# Ensemble on validation
# =========================

val_pred_cat = final_cat.predict(X_val)
val_pred_lgb = final_lgb.predict(X_val)

val_pred_ens = 0.5 * val_pred_cat + 0.5 * val_pred_lgb

ens_thresholds, ens_val_qwk = optimize_thresholds(y_val, val_pred_ens)

print("Ensemble thresholds:", ens_thresholds)
print("Ensemble val QWK  :", ens_val_qwk)

# =========================
# Final training (CatBoost + LGBM already on full train)
# =========================

# =========================
# Predictions for test + submission
# =========================

test_pred_cat = final_cat.predict(X_test)
test_pred_lgb = final_lgb.predict(X_test)

test_pred_ens = 0.5 * test_pred_cat + 0.5 * test_pred_lgb

# Use ensemble thresholds
t1, t2, t3 = ens_thresholds
test_cls = np.digitize(test_pred_ens, [t1, t2, t3]) + 1  # 1..4

submission = pd.DataFrame({
    "PetID": test["PetID"],
    "AdoptionSpeed": test_cls.astype(int)
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to: {SUBMISSION_PATH}")
print(submission.head())
