import os
import re
import gc
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
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

# =========================
# Config
# =========================

DATA_DIR = "C:/Goit/goit-mldl-hw-14/data/petfinder/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

TRAIN_IMG_DIR = "./data/petfinder/images/images/train"
TEST_IMG_DIR  = "./data/petfinder/images/images/test"

TEXT_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "efficientnet_b0"

IMAGE_SIZE = 224
BATCH_SIZE_IMG = 32   # safe for MX550

TFIDF_MAX_FEATURES = 500
COLOR_BINS = 32

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
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def optimize_thresholds(y_true, y_pred_cont):
    y_true = np.array(y_true, dtype=int)
    y_pred_cont = np.array(y_pred_cont, dtype=float)

    qs = np.percentile(y_pred_cont, [25, 50, 75])
    best_t = qs.copy()
    best_kappa = -1.0

    for _ in range(300):
        t1 = best_t[0] + np.random.uniform(-0.3, 0.3)
        t2 = best_t[1] + np.random.uniform(-0.3, 0.3)
        t3 = best_t[2] + np.random.uniform(-0.3, 0.3)
        if not (t1 < t2 < t3):
            continue

        y_pred_cls = np.digitize(y_pred_cont, [t1, t2, t3]) + 1
        kappa = quadratic_weighted_kappa(y_true, y_pred_cls)
        if kappa > best_kappa:
            best_kappa = kappa
            best_t = np.array([t1, t2, t3])

    return best_t, best_kappa

# =========================
# Load data
# =========================

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

train["Description"] = train["Description"].fillna("").apply(clean_text)
test["Description"] = test["Description"].fillna("").apply(clean_text)

# =========================
# Photo stats
# =========================

train_counts = Counter()
for f in os.listdir(TRAIN_IMG_DIR):
    if f.endswith(".jpg"):
        pet_id = f.split("-")[0]
        train_counts[pet_id] += 1

# =========================
# SentenceTransformer embeddings
# =========================

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

# =========================
# Image embeddings (3 photos per pet)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

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

        candidates = sorted([
            f for f in os.listdir(self.img_dir)
            if f.startswith(pet_id) and f.endswith(".jpg")
        ])

        candidates = candidates[:3]

        imgs = []
        if len(candidates) == 0:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            imgs = [transform_img(img)] * 3
        else:
            for fname in candidates:
                img_path = os.path.join(self.img_dir, fname)
                img = Image.open(img_path).convert("RGB")
                imgs.append(transform_img(img))

            while len(imgs) < 3:
                imgs.append(imgs[-1])

        return torch.stack(imgs, dim=0)


def compute_image_embeddings(df, img_dir):
    dataset = PetImageDataset(df, img_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_IMG, shuffle=False)

    all_emb = []
    with torch.no_grad():
        for batch in loader:
            B, N, C, H, W = batch.shape
            batch = batch.view(B * N, C, H, W).to(device)

            emb = image_model(batch)
            emb = emb.view(B, N, -1)
            emb = emb.mean(dim=1)

            all_emb.append(emb.cpu().numpy())

    return np.concatenate(all_emb, axis=0)

train_img_emb = compute_image_embeddings(train, TRAIN_IMG_DIR)
test_img_emb  = compute_image_embeddings(test, TEST_IMG_DIR)

# =========================
# TF-IDF
# =========================

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2
)
tfidf_train = tfidf.fit_transform(train["Description"].tolist()).toarray()
tfidf_test = tfidf.transform(test["Description"].tolist()).toarray()

# =========================
# Color histograms
# =========================

def compute_color_histograms(df, img_dir, bins=COLOR_BINS):
    hist_list = []
    for pet_id in df["PetID"].tolist():
        candidates = sorted([
            f for f in os.listdir(img_dir)
            if f.startswith(pet_id) and f.endswith(".jpg")
        ])
        if len(candidates) == 0:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        else:
            img_path = os.path.join(img_dir, candidates[0])
            img = Image.open(img_path).convert("RGB")

        img_np = np.array(img)
        hists = []
        for c in range(3):
            channel = img_np[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=bins, range=(0, 255), density=True)
            hists.append(hist)
        hist_list.append(np.concatenate(hists))

    return np.array(hist_list)

train_color = compute_color_histograms(train, TRAIN_IMG_DIR)
test_color  = compute_color_histograms(test, TEST_IMG_DIR)

# =========================
# Photo count feature
# =========================

def get_photo_counts(df, counter_map):
    return np.array([counter_map.get(pid, 0) for pid in df["PetID"]]).reshape(-1, 1)

train_photo_counts = get_photo_counts(train, train_counts)
test_photo_counts  = get_photo_counts(test, train_counts)

# =========================
# PCA
# =========================

pca_img = PCA(n_components=PCA_IMG_DIM, random_state=RANDOM_SEED)
pca_txt = PCA(n_components=PCA_TXT_DIM, random_state=RANDOM_SEED)

train_img_pca = pca_img.fit_transform(train_img_emb)
test_img_pca  = pca_img.transform(test_img_emb)

train_txt_pca = pca_txt.fit_transform(train_text_emb)
test_txt_pca  = pca_txt.transform(test_text_emb)

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

# =========================
# Train/val split (for thresholds & QWK)
# =========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_train
)

# =========================
# Optuna + CatBoost (on split)
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

import optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

best_params = study.best_params

# =========================
# Final CatBoost (on full train)
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

# =========================
# LightGBM (on full train)
# =========================

final_lgb = lgb.LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=RANDOM_SEED,
    n_estimators=600
)

final_lgb.fit(X_train, y_train)

# =========================
# Threshold optimization WITHOUT leakage (use train split)
# =========================

train_pred_cat = final_cat.predict(X_tr)
cat_thr, _ = optimize_thresholds(y_tr, train_pred_cat)

train_pred_lgb = final_lgb.predict(X_tr)
lgb_thr, _ = optimize_thresholds(y_tr, train_pred_lgb)

train_pred_ens = 0.5 * train_pred_cat + 0.5 * train_pred_lgb
ens_thr, _ = optimize_thresholds(y_tr, train_pred_ens)

# =========================
# Evaluate on VAL split (real QWK)
# =========================

val_pred_cat = final_cat.predict(X_val)
val_cat_cls = np.digitize(val_pred_cat, cat_thr) + 1
cat_val_qwk = quadratic_weighted_kappa(y_val, val_cat_cls)
print("CatBoost val QWK (real):", cat_val_qwk)

val_pred_lgb = final_lgb.predict(X_val)
val_lgb_cls = np.digitize(val_pred_lgb, lgb_thr) + 1
lgb_val_qwk = quadratic_weighted_kappa(y_val, val_lgb_cls)
print("LightGBM val QWK (real):", lgb_val_qwk)

val_pred_ens = 0.5 * val_pred_cat + 0.5 * val_pred_lgb
val_ens_cls = np.digitize(val_pred_ens, ens_thr) + 1
ens_val_qwk = quadratic_weighted_kappa(y_val, val_ens_cls)
print("Ensemble thresholds:", ens_thr)
print("Ensemble val QWK (real):", ens_val_qwk)

# =========================
# Test predictions + submission
# =========================

test_pred_cat = final_cat.predict(X_test)
test_pred_lgb = final_lgb.predict(X_test)
test_pred_ens = 0.5 * test_pred_cat + 0.5 * test_pred_lgb

test_ens_cls = np.digitize(test_pred_ens, ens_thr) + 1

submission = pd.DataFrame({
    "PetID": test["PetID"].values,
    "AdoptionSpeed": test_ens_cls
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission: {SUBMISSION_PATH}")
print(submission.head())
