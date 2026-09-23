import os
import gc
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm

import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from datasets import load_dataset

import random
import copy
import time
import shutil

import zipfile

from sklearn.metrics import confusion_matrix, classification_report, f1_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from collections import Counter

from pathlib import Path

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

from catboost import CatBoostRegressor

import sentence_transformers
from sentence_transformers import SentenceTransformer

import timm

from PIL import Image


# Завантаження даних

# VSCode path definition
data_dir = "./data/petfinder/"

# Завантаження train/test через ImageFolder
train_dir = os.path.join(data_dir, "images/images/train")
test_dir  = os.path.join(data_dir, "images/images/test")

TRAIN_CSV = data_dir + "train.csv"
TEST_CSV = data_dir + "test.csv"

TRAIN_IMAGES = train_dir
TEST_IMAGES = test_dir

print("Train dir exists:", os.path.exists(train_dir))
print("Test dir exists:", os.path.exists(test_dir))

TRAIN_IMG_DIR = TRAIN_IMAGES
TEST_IMG_DIR = TEST_IMAGES

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

# Перевірка 1: Загальна інформація
print("\nПеревірка 1: Загальна інформація")
print("TRAIN:", train.shape)
print("TEST :", test.shape)

print("\nColumns:")
print(train.columns.tolist())

train.head()

# Перевірка 2: Типи колонок
print("\nПеревірка 2: Типи колонок")
print(train.dtypes)

# Перевірка 3: Пропуски
print("\nПеревірка 3: Пропуски")
na = train.isnull().sum()

print(
    na[na > 0]
    .sort_values(ascending=False)
)


#Перевірка цільової змінної

train["AdoptionSpeed"].value_counts().sort_index()

# Текстове очищення

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# Перевірка 4: Цільова змінна
print("\nПеревірка 4: Цільова змінна")
print(
    train["AdoptionSpeed"]
    .value_counts()
    .sort_index()
)


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

# Перевірка 5: Опис табличних ознак
print("\nПеревірка 5: Опис табличних ознак")
print(
    train.describe(
    include="all"
    ).T
)

# Перевірка 6: Перевірка текстів
#print("\nПеревірка 6: Перевірка текстів")
#print(
#    train["Description"]
#    .head(10)
#    .tolist()
#)

# Перевірка 7: Довжина текстів
print("\nПеревірка 7: Довжина текстів")
desc_len = (
    train["Description"]
    .fillna("")
    .str.len()
)

print(desc_len.describe())

# Перевірка 8: Кількість фотографій
print("\nПеревірка 8: Кількість фотографій")
from collections import Counter

train_counts = Counter()

for f in os.listdir(TRAIN_IMAGES):
    if f.endswith(".jpg"):
        pet_id = f.split("-")[0]
        train_counts[pet_id] += 1

vals = list(train_counts.values())

print("Min:", min(vals))
print("Mean:", np.mean(vals))
print("Max:", max(vals))
print("Median:", np.median(vals))

# Перевірка 9: Розподіл фото
print("\nПеревірка 9: Розподіл фото")

photo_hist = Counter(vals)

for k, v in sorted(photo_hist.items()):
    print(k, "photos ->", v)


# Text Embeddings (MiniLM)

model_text = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

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

# Перевірка 10: Перевірка embedding
print("\nПеревірка 10.1: Перевірка embedding")
print(train_text_emb.shape)
print(test_text_emb.shape)


# ==========================
# 11. Image embeddings (EfficientNet-B0)
# ==========================

# Обережно з CUDA: якщо немає GPU, падаємо на CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

image_model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0  # повертає фічі, а не логіти
)

image_model.eval()
image_model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Мапа: PetID -> список шляхів до фото
def build_image_index(img_dir):
    img_index = {}
    for f in os.listdir(img_dir):
        if not f.endswith(".jpg"):
            continue
        pet_id = f.split("-")[0]
        path = os.path.join(img_dir, f)
        img_index.setdefault(pet_id, []).append(path)
    return img_index

train_img_index = build_image_index(TRAIN_IMAGES)
test_img_index  = build_image_index(TEST_IMAGES)

print("Train image index size:", len(train_img_index))
print("Test image index size:", len(test_img_index))

# Функція для отримання embedding по одному PetID
def get_image_embedding_for_pet(pet_id, img_index, model, device, transform):
    paths = img_index.get(pet_id, [])
    if len(paths) == 0:
        # якщо немає фото, повертаємо нульовий вектор
        return np.zeros(1280, dtype=np.float32)

    # для простоти беремо перше фото (можна усереднювати всі)
    path = paths[0]
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return np.zeros(1280, dtype=np.float32)

    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img_t)
    emb = emb.squeeze().cpu().numpy().astype(np.float32)
    return emb

# Обчислюємо image embeddings для train/test
train_img_emb_list = []
for pet_id in tqdm(train["PetID"].tolist(), desc="Train image embeddings"):
    emb = get_image_embedding_for_pet(
        pet_id,
        train_img_index,
        image_model,
        device,
        transform
    )
    train_img_emb_list.append(emb)

test_img_emb_list = []
for pet_id in tqdm(test["PetID"].tolist(), desc="Test image embeddings"):
    emb = get_image_embedding_for_pet(
        pet_id,
        test_img_index,
        image_model,
        device,
        transform
    )
    test_img_emb_list.append(emb)

train_img_emb = np.stack(train_img_emb_list)
test_img_emb  = np.stack(test_img_emb_list)

print("Train image emb shape:", train_img_emb.shape)
print("Test image emb shape:", test_img_emb.shape)


# ==========================
# 12. Простий табличний фічерінг
# ==========================

# Довжина опису
train_desc_len = (
    train["Description"]
    .fillna("")
    .str.len()
    .astype(np.float32)
).values.reshape(-1, 1)

test_desc_len = (
    test["Description"]
    .fillna("")
    .str.len()
    .astype(np.float32)
).values.reshape(-1, 1)

# Кількість фото
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
# 13. Збирання фінальних фічей
# ==========================

# У тебе вже є:
# train_text_emb, test_text_emb (MiniLM, розмір 384)
# train_img_emb, test_img_emb (EfficientNet-B0, розмір 1280)
# train_desc_len, test_desc_len (1)
# train_photo_cnt, test_photo_cnt (1)

X_train = np.concatenate(
    [train_text_emb, train_img_emb, train_desc_len, train_photo_cnt],
    axis=1
)
X_test = np.concatenate(
    [test_text_emb, test_img_emb, test_desc_len, test_photo_cnt],
    axis=1
)

y_train = train["AdoptionSpeed"].values.astype(np.int32)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)


# ==========================
# 14. Train/validation split
# ==========================

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

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
# 15. CatBoost baseline (Regressor)
# ==========================

from catboost import CatBoostRegressor

cat_params = {
    "loss_function": "RMSE",
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 1000,
    "random_seed": 42,
    "verbose": 100,
    "task_type": "CPU" # if device <> "cuda" else "GPU"
}

model_cb = CatBoostRegressor(**cat_params)

model_cb.fit(
    X_tr,
    y_tr,
    eval_set=(X_val, y_val),
    use_best_model=True
)


# ==========================
# 16. Quadratic Weighted Kappa + threshold optimization
# ==========================

def qwk(y_true, y_pred_labels):
    return cohen_kappa_score(y_true, y_pred_labels, weights="quadratic")

# CatBoostRegressor повертає float-прогнози, треба перетворити на класи 1..4
# Будемо оптимізувати 3 пороги: t1, t2, t3
# Класи:
# <= t1 -> 1
# <= t2 -> 2
# <= t3 -> 3
# >  t3 -> 4

def apply_thresholds(preds, thresholds):
    t1, t2, t3 = thresholds
    labels = np.zeros_like(preds, dtype=np.int32)
    labels[preds <= t1] = 1
    labels[(preds > t1) & (preds <= t2)] = 2
    labels[(preds > t2) & (preds <= t3)] = 3
    labels[preds > t3] = 4
    return labels

# Грубий пошук порогів по валідації
val_preds_reg = model_cb.predict(X_val)

def optimize_thresholds(y_true, preds):
    best_kappa = -1.0
    best_thr = None

    # Діапазон беремо з предиктів
    p_min, p_max = preds.min(), preds.max()
    grid = np.linspace(p_min, p_max, 50)

    for t1 in grid:
        for t2 in grid:
            if t2 <= t1:
                continue
            for t3 in grid:
                if t3 <= t2:
                    continue
                thr = (t1, t2, t3)
                labels = apply_thresholds(preds, thr)
                kappa = qwk(y_true, labels)
                if kappa > best_kappa:
                    best_kappa = kappa
                    best_thr = thr
    return best_thr, best_kappa

print("Optimizing thresholds on validation...")
best_thr, best_kappa = optimize_thresholds(y_val, val_preds_reg)
print("Best thresholds:", best_thr)
print("Best val QWK   :", best_kappa)


# ==========================
# 17. Фінальне донавчання на всьому train
# ==========================

# Перенавчаємо CatBoost на всіх доступних даних (train)
final_cb = CatBoostRegressor(**cat_params)
final_cb.fit(
    X_train,
    y_train,
    verbose=100
)


# ==========================
# 18. Прогнози для test + submission.csv
# ==========================

test_preds_reg = final_cb.predict(X_test)
test_preds_cls = apply_thresholds(test_preds_reg, best_thr)

submission = pd.DataFrame({
    "PetID": test["PetID"],
    "AdoptionSpeed": test_preds_cls
})

sub_path = os.path.join(data_dir, "submission.csv")
submission.to_csv(sub_path, index=False)

print("Saved submission to:", sub_path)
print(submission.head())
