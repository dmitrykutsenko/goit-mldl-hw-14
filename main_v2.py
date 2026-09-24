import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import torch
import torch.nn as nn
from torchvision import transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from scipy.optimize import minimize
from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
import timm
from PIL import Image


# ==========================
# 0. Paths and data loading
# ==========================

data_dir = "./data/petfinder/"

train_dir = os.path.join(data_dir, "images/images/train")
test_dir  = os.path.join(data_dir, "images/images/test")

TRAIN_CSV = data_dir + "train.csv"
TEST_CSV  = data_dir + "test.csv"

print("Train dir exists:", os.path.exists(train_dir))
print("Test dir exists:", os.path.exists(test_dir))

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

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train["Description"] = train["Description"].fillna("").apply(clean_text)
test["Description"]  = test["Description"].fillna("").apply(clean_text)

print("\nПеревірка 5: Опис табличних ознак")
print(train.describe(include="all").T)

print("\nПеревірка 7: Довжина текстів")
desc_len = train["Description"].fillna("").str.len()
print(desc_len.describe())

print("\nПеревірка 8: Кількість фотографій")
train_counts = Counter()
for f in os.listdir(train_dir):
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


# ==========================
# 1. Text embeddings (MiniLM)
# ==========================

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


# ==========================
# 2. Image embeddings (EfficientNet-B0)
# ==========================

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

def build_image_index(img_dir):
    img_index = {}
    for f in os.listdir(img_dir):
        if not f.endswith(".jpg"):
            continue
        pet_id = f.split("-")[0]
        path = os.path.join(img_dir, f)
        img_index.setdefault(pet_id, []).append(path)
    return img_index

train_img_index = build_image_index(train_dir)
test_img_index  = build_image_index(test_dir)

print("Train image index size:", len(train_img_index))
print("Test image index size:", len(test_img_index))

def get_image_embedding(path, model, device, transform):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return np.zeros(1280, dtype=np.float32)
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_t)
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    return emb

def get_image_embedding_for_pet(pet_id, img_index, model, device, transform):
    paths = img_index.get(pet_id, [])
    if len(paths) == 0:
        return np.zeros(1280, dtype=np.float32)
    # можна усереднювати всі фото, але для швидкості беремо перше
    return get_image_embedding(paths[0], model, device, transform)

train_img_emb_list = []
for pet_id in tqdm(train["PetID"].tolist(), desc="Train image embeddings"):
    emb = get_image_embedding_for_pet(
        pet_id, train_img_index, image_model, device, img_transform
    )
    train_img_emb_list.append(emb)

test_img_emb_list = []
for pet_id in tqdm(test["PetID"].tolist(), desc="Test image embeddings"):
    emb = get_image_embedding_for_pet(
        pet_id, test_img_index, image_model, device, img_transform
    )
    test_img_emb_list.append(emb)

train_img_emb = np.stack(train_img_emb_list)
test_img_emb  = np.stack(test_img_emb_list)

print("Train image emb shape:", train_img_emb.shape)
print("Test image emb shape:", test_img_emb.shape)


# ==========================
# 3. TF-IDF text features
# ==========================

tfidf = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=3
)

tfidf_train = tfidf.fit_transform(train["Description"].tolist())
tfidf_test  = tfidf.transform(test["Description"].tolist())

tfidf_train = tfidf_train.astype(np.float32)
tfidf_test  = tfidf_test.astype(np.float32)

print("TF-IDF shapes:", tfidf_train.shape, tfidf_test.shape)


# ==========================
# 4. Color histograms (R,G,B)
# ==========================

def get_color_hist_for_pet(pet_id, img_index, bins=32):
    paths = img_index.get(pet_id, [])
    if len(paths) == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    path = paths[0]
    try:
        img = cv2.imread(path)
        if img is None:
            return np.zeros(bins * 3, dtype=np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.zeros(bins * 3, dtype=np.float32)
    hist_features = []
    for c in range(3):
        channel = img[:, :, c]
        hist, _ = np.histogram(channel, bins=bins, range=(0, 255))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        hist_features.append(hist)
    return np.concatenate(hist_features)

train_color_list = []
for pet_id in tqdm(train["PetID"].tolist(), desc="Train color hist"):
    h = get_color_hist_for_pet(pet_id, train_img_index, bins=32)
    train_color_list.append(h)

test_color_list = []
for pet_id in tqdm(test["PetID"].tolist(), desc="Test color hist"):
    h = get_color_hist_for_pet(pet_id, test_img_index, bins=32)
    test_color_list.append(h)

train_color = np.stack(train_color_list)
test_color  = np.stack(test_color_list)

print("Color hist shapes:", train_color.shape, test_color.shape)


# ==========================
# 5. Simple tabular features
# ==========================

train_desc_len = (
    train["Description"].fillna("").str.len().astype(np.float32)
).values.reshape(-1, 1)

test_desc_len = (
    test["Description"].fillna("").str.len().astype(np.float32)
).values.reshape(-1, 1)

def get_photo_count_series(df, img_index):
    counts = []
    for pet_id in df["PetID"].tolist():
        counts.append(len(img_index.get(pet_id, [])))
    return np.array(counts, dtype=np.float32).reshape(-1, 1)

train_photo_cnt = get_photo_count_series(train, train_img_index)
test_photo_cnt  = get_photo_count_series(test, test_img_index)

print("Train photo count stats:",
      np.min(train_photo_cnt),
      np.mean(train_photo_cnt),
      np.max(train_photo_cnt))


# ==========================
# 6. Optional PCA (to keep size reasonable)
# ==========================

# Для max accuracy можна залишити повні ембедінги,
# але щоб не роздувати моделі, трохи стискаємо image/text.

pca_img = PCA(n_components=256, random_state=42)
train_img_emb_pca = pca_img.fit_transform(train_img_emb)
test_img_emb_pca  = pca_img.transform(test_img_emb)

pca_text = PCA(n_components=128, random_state=42)
train_text_emb_pca = pca_text.fit_transform(train_text_emb)
test_text_emb_pca  = pca_text.transform(test_text_emb)

print("PCA image:", train_img_emb_pca.shape, test_img_emb_pca.shape)
print("PCA text :", train_text_emb_pca.shape, test_text_emb_pca.shape)


# ==========================
# 7. Final feature matrix
# ==========================

# TF-IDF у форматі sparse → конвертуємо в dense для CatBoost/LightGBM
tfidf_train_dense = tfidf_train.toarray()
tfidf_test_dense  = tfidf_test.toarray()

X_train = np.concatenate(
    [
        train_text_emb_pca,
        train_img_emb_pca,
        tfidf_train_dense,
        train_color,
        train_desc_len,
        train_photo_cnt,
    ],
    axis=1
)

X_test = np.concatenate(
    [
        test_text_emb_pca,
        test_img_emb_pca,
        tfidf_test_dense,
        test_color,
        test_desc_len,
        test_photo_cnt,
    ],
    axis=1
)

y_train = train["AdoptionSpeed"].values.astype(np.int32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)


# ==========================
# 8. Train/validation split
# ==========================

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Train split:", X_tr.shape, y_tr.shape)
print("Val split  :", X_val.shape, y_val.shape)


# ==========================
# 9. QWK + threshold optimizer (Nelder–Mead)
# ==========================

def qwk(y_true, y_pred_labels):
    return cohen_kappa_score(y_true, y_pred_labels, weights="quadratic")

def apply_thresholds(preds, thresholds):
    # thresholds: [t1, t2, t3]
    t1, t2, t3 = thresholds
    labels = np.zeros_like(preds, dtype=np.int32)
    labels[preds <= t1] = 1
    labels[(preds > t1) & (preds <= t2)] = 2
    labels[(preds > t2) & (preds <= t3)] = 3
    labels[preds > t3] = 4
    return labels

def optimize_thresholds(y_true, preds):
    # стартові пороги — приблизно між класами
    initial = np.array([1.5, 2.5, 3.5], dtype=np.float64)

    def loss(thr):
        thr_sorted = np.sort(thr)
        labels = apply_thresholds(preds, thr_sorted)
        return -qwk(y_true, labels)

    result = minimize(
        loss,
        initial,
        method="nelder-mead",
        options={"maxiter": 200, "disp": False}
    )
    best_thr = np.sort(result.x)
    best_kappa = -result.fun
    return best_thr, best_kappa


# ==========================
# 10. Optuna for CatBoost
# ==========================

def catboost_objective(trial):
    params = {
        "loss_function": "RMSE",
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "iterations": trial.suggest_int("iterations", 400, 1200),
        "random_seed": 42,
        "verbose": False,
        "task_type": "CPU",
    }

    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

    preds_val = model.predict(X_val)
    thr, kappa = optimize_thresholds(y_val, preds_val)
    return kappa

print("Running Optuna for CatBoost (this може зайняти кілька хвилин)...")
study_cb = optuna.create_study(direction="maximize")
study_cb.optimize(catboost_objective, n_trials=30)

best_params_cb = study_cb.best_trial.params
print("Best CatBoost params:", best_params_cb)
print("Best CatBoost val QWK:", study_cb.best_value)

# Фіксуємо повний набір параметрів
cat_params = {
    "loss_function": "RMSE",
    "depth": best_params_cb["depth"],
    "learning_rate": best_params_cb["learning_rate"],
    "l2_leaf_reg": best_params_cb["l2_leaf_reg"],
    "iterations": best_params_cb["iterations"],
    "random_seed": 42,
    "verbose": 100,
    "task_type": "CPU",
}

model_cb = CatBoostRegressor(**cat_params)
model_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

val_preds_cb = model_cb.predict(X_val)
best_thr_cb, best_kappa_cb = optimize_thresholds(y_val, val_preds_cb)
print("CatBoost thresholds:", best_thr_cb)
print("CatBoost val QWK  :", best_kappa_cb)


# ==========================
# 11. LightGBM model
# ==========================

lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}

callbacks = [lgb.early_stopping(stopping_rounds=100)]

model_lgb = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    valid_names=["val"]
)

val_preds_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)


# ==========================
# 12. Ensemble (CatBoost + LightGBM)
# ==========================

val_preds_ensemble = 0.5 * val_preds_cb + 0.5 * val_preds_lgb
best_thr_ens, best_kappa_ens = optimize_thresholds(y_val, val_preds_ensemble)

print("Ensemble thresholds:", best_thr_ens)
print("Ensemble val QWK  :", best_kappa_ens)


# ==========================
# 13. Final training on all train
# ==========================

# Перенавчаємо CatBoost на всіх даних
final_cb = CatBoostRegressor(**cat_params)
final_cb.fit(X_train, y_train, verbose=100)

# Перенавчаємо LightGBM на всіх даних
lgb_full_train = lgb.Dataset(X_train, label=y_train)
final_lgb = lgb.train(
    lgb_params,
    lgb_full_train,
    num_boost_round=model_lgb.best_iteration or 1000,
    verbose_eval=False
)


# ==========================
# 14. Predictions for test + submission
# ==========================

test_preds_cb  = final_cb.predict(X_test)
test_preds_lgb = final_lgb.predict(X_test)

test_preds_ens = 0.5 * test_preds_cb + 0.5 * test_preds_lgb

test_preds_cls = apply_thresholds(test_preds_ens, best_thr_ens)

submission = pd.DataFrame({
    "PetID": test["PetID"],
    "AdoptionSpeed": test_preds_cls
})

sub_path = os.path.join(data_dir, "submission.csv")
submission.to_csv(sub_path, index=False)

print("Saved submission to:", sub_path)
print(submission.head())
