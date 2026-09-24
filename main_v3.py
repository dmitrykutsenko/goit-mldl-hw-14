import os
import gc
import re
import time
import random
import copy

import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
import timm

from PIL import Image

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from catboost import CatBoostRegressor

import optuna


# =========================
# 1. Шлях до даних
# =========================

DATA_DIR = "./data/petfinder/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images/images/train")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "images/images/test")

print("Train dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test dir exists:", os.path.exists(TEST_IMG_DIR))

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

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


# =========================
# 2. Очищення тексту
# =========================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train["Description"] = (
    train["Description"]
    .fillna("")
    .apply(clean_text)
)

test["Description"] = (
    test["Description"]
    .fillna("")
    .apply(clean_text)
)

print("\nПеревірка 7: Довжина текстів")
desc_len = train["Description"].str.len()
print(desc_len.describe())


# =========================
# 3. Статистика фото
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
# 4. Text embeddings (MiniLM)
# =========================

print("\nЗавантаження SentenceTransformer (MiniLM)...")
model_text = SentenceTransformer("all-MiniLM-L6-v2")

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
# 5. Image embeddings (EfficientNet-B0)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

image_model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0
)
image_model.eval()
image_model.to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_image_embedding(pet_id, img_dir):
    # беремо перше фото для кожного PetID
    files = [f for f in os.listdir(img_dir) if f.startswith(pet_id) and f.endswith(".jpg")]
    if len(files) == 0:
        # якщо немає фото — повертаємо нулі
        return np.zeros(1280, dtype=np.float32)
    img_path = os.path.join(img_dir, files[0])
    try:
        img = Image.open(img_path).convert("RGB")
        img = img_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = image_model(img).cpu().numpy().reshape(-1)
        return emb
    except Exception:
        return np.zeros(1280, dtype=np.float32)

def build_image_embeddings(df, img_dir, label):
    embs = []
    pet_ids = df["PetID"].tolist()
    print(f"{label} image index size:", len(pet_ids))
    for pid in pet_ids:
        embs.append(load_image_embedding(pid, img_dir))
    embs = np.stack(embs, axis=0)
    return embs

print("\nОбчислення image embeddings...")
train_img_emb = build_image_embeddings(train, TRAIN_IMG_DIR, "Train")
test_img_emb  = build_image_embeddings(test,  TEST_IMG_DIR,  "Test")

print("Train image emb shape:", train_img_emb.shape)
print("Test image emb shape:",  test_img_emb.shape)


# =========================
# 6. TF-IDF для тексту
# =========================

print("\nОбчислення TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2
)

tfidf_train = tfidf.fit_transform(train["Description"])
tfidf_test  = tfidf.transform(test["Description"])

tfidf_train = tfidf_train.toarray().astype(np.float32)
tfidf_test  = tfidf_test.toarray().astype(np.float32)

print("TF-IDF shapes:", tfidf_train.shape, tfidf_test.shape)


# =========================
# 7. Color histograms
# =========================

print("\nОбчислення color histograms...")

def compute_color_hist(pet_id, img_dir, bins=32):
    files = [f for f in os.listdir(img_dir) if f.startswith(pet_id) and f.endswith(".jpg")]
    if len(files) == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    img_path = os.path.join(img_dir, files[0])
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        hist_r, _ = np.histogram(arr[:, :, 0], bins=bins, range=(0, 255), density=True)
        hist_g, _ = np.histogram(arr[:, :, 1], bins=bins, range=(0, 255), density=True)
        hist_b, _ = np.histogram(arr[:, :, 2], bins=bins, range=(0, 255), density=True)
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
        return hist
    except Exception:
        return np.zeros(bins * 3, dtype=np.float32)

def build_color_hists(df, img_dir, label):
    hists = []
    pet_ids = df["PetID"].tolist()
    print(f"{label} color hist: {len(pet_ids)}")
    for pid in pet_ids:
        hists.append(compute_color_hist(pid, img_dir))
    hists = np.stack(hists, axis=0)
    return hists

train_color = build_color_hists(train, TRAIN_IMG_DIR, "Train")
test_color  = build_color_hists(test,  TEST_IMG_DIR,  "Test")

print("Color hist shapes:", train_color.shape, test_color.shape)


# =========================
# 8. Додаткові табличні фічі
# =========================

print("\nTrain photo count stats:")
photo_count_map = train_counts
train_photo_count = train["PetID"].map(lambda x: photo_count_map.get(x, 0)).astype(np.float32)
test_photo_count  = test["PetID"].map(lambda x: photo_count_map.get(x, 0)).astype(np.float32)

print(train_photo_count.min(), train_photo_count.mean(), train_photo_count.max())


# =========================
# 9. PCA для image/text
# =========================

print("\nPCA для image/text...")

pca_img = PCA(n_components=256, random_state=42)
pca_txt = PCA(n_components=128, random_state=42)

pca_img_train = pca_img.fit_transform(train_img_emb)
pca_img_test  = pca_img.transform(test_img_emb)

pca_txt_train = pca_txt.fit_transform(train_text_emb)
pca_txt_test  = pca_txt.transform(test_text_emb)

print("PCA image:", pca_img_train.shape, pca_img_test.shape)
print("PCA text :", pca_txt_train.shape, pca_txt_test.shape)


# =========================
# 10. Збір фінальних фіч
# =========================

X_train = np.concatenate([
    pca_img_train,
    pca_txt_train,
    tfidf_train,
    train_color,
    train_photo_count.values.reshape(-1, 1)
], axis=1)

X_test = np.concatenate([
    pca_img_test,
    pca_txt_test,
    tfidf_test,
    test_color,
    test_photo_count.values.reshape(-1, 1)
], axis=1)

y_train = train["AdoptionSpeed"].values.astype(np.float32)

print("X_train shape:", X_train.shape)
print("X_test shape:",  X_test.shape)
print("y_train shape:", y_train.shape)


# =========================
# 11. Train/Val split
# =========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Train split:", X_tr.shape, y_tr.shape)
print("Val split  :", X_val.shape, y_val.shape)


# =========================
# 12. QWK метрика
# =========================

def quadratic_weighted_kappa(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    min_rating = 0
    max_rating = 4

    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = np.zeros((num_ratings, num_ratings), dtype=np.float64)

    for a, p in zip(y_true, y_pred):
        conf_mat[a, p] += 1.0

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_pred = np.bincount(y_pred, minlength=num_ratings)

    E = np.outer(hist_true, hist_pred) / y_true.shape[0]

    W = np.zeros((num_ratings, num_ratings), dtype=np.float64)
    for i in range(num_ratings):
        for j in range(num_ratings):
            W[i, j] = ((i - j) ** 2) / ((max_rating - min_rating) ** 2)

    numerator = np.sum(W * conf_mat)
    denominator = np.sum(W * E)
    return 1.0 - numerator / denominator


# =========================
# 13. Оптимізація порогів
# =========================

def optimize_thresholds(y_true, y_pred_cont):
    from scipy.optimize import minimize

    def _loss(ths):
        t1, t2, t3 = ths
        y_pred = np.digitize(y_pred_cont, bins=[t1, t2, t3])
        return -quadratic_weighted_kappa(y_true, y_pred)

    x0 = [1.5, 2.5, 3.5]
    res = minimize(_loss, x0, method="Nelder-Mead")
    return res.x


# =========================
# 14. Optuna для CatBoost
# =========================

def objective_catboost(trial):
    depth = trial.suggest_int("depth", 4, 8)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
    iterations = trial.suggest_int("iterations", 500, 1200)

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        random_seed=42,
        verbose=False,
        #task_type="GPU" if torch.cuda.is_available() else "CPU"
        task_type="CPU"
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=False
    )

    val_pred_cont = model.predict(X_val)
    ths = optimize_thresholds(y_val, val_pred_cont)
    val_pred_disc = np.digitize(val_pred_cont, bins=ths)
    qwk = quadratic_weighted_kappa(y_val, val_pred_disc)
    return qwk

print("\nRunning Optuna for CatBoost (це може зайняти час)...")
study = optuna.create_study(direction="maximize")
study.optimize(objective_catboost, n_trials=30)

best_params = study.best_params
best_qwk = study.best_value

print("Best CatBoost params:", best_params)
print("Best CatBoost val QWK:", best_qwk)


# =========================
# 15. Фінальний CatBoost на всіх даних
# =========================

final_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=best_params["depth"],
    learning_rate=best_params["learning_rate"],
    l2_leaf_reg=best_params["l2_leaf_reg"],
    iterations=best_params["iterations"],
    random_seed=42,
    verbose=False,
    #task_type="GPU" if torch.cuda.is_available() else "CPU"
    task_type="CPU"
)

final_model.fit(
    X_train, y_train,
    verbose=False
)

# пороги оптимізуємо на валідації (X_val, y_val)
val_pred_cont = final_model.predict(X_val)
final_ths = optimize_thresholds(y_val, val_pred_cont)
val_pred_disc = np.digitize(val_pred_cont, bins=final_ths)
final_qwk = quadratic_weighted_kappa(y_val, val_pred_disc)

print("Final thresholds:", final_ths)
print("Final val QWK  :", final_qwk)


# =========================
# 16. Прогнози на тесті + submission.csv
# =========================

test_pred_cont = final_model.predict(X_test)
test_pred_disc = np.digitize(test_pred_cont, bins=final_ths)

# обмежуємо класи 0..4 (на випадок виходу за межі)
test_pred_disc = np.clip(test_pred_disc, 0, 4)

submission = pd.DataFrame({
    "PetID": test["PetID"],
    "AdoptionSpeed": test_pred_disc.astype(int)
})

out_path = os.path.join(DATA_DIR, "submission_catboost_only.csv")
submission.to_csv(out_path, index=False)

print(f"Saved submission to: {out_path}")
print(submission.head())
