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
print("\nПеревірка 8: Кількість фотографій")

photo_hist = Counter(vals)

for k, v in sorted(photo_hist.items()):
    print(k, "photos ->", v)


# Text Embeddings (MiniLM)