# ============================================
# MODIS FIRMS INDONESIA - FINAL PIPELINE
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =====================================================
# OUTPUT FOLDER
# =====================================================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(f"{OUTPUT_DIR}/{name}", dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================
# LOAD DATASET
# =====================================================
df_list = []

for year in range(2020,2025):
    df_list.append(pd.read_csv(f"dataset/modis_{year}_Indonesia.csv"))

data = pd.concat(df_list, ignore_index=True)

print("Initial Shape :", data.shape)

# Backup data sebelum cleaning
data_before_cleaning = data.copy()

# dictionary untuk menyimpan alasan cleaning
cleaning_log = {}


# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["acq_date"] = data["acq_date"].astype(str)

data[["Year","Month","Day"]] = data["acq_date"].str.split("-", expand=True)

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
data["Day"] = pd.to_numeric(data["Day"], errors="coerce")

data["acq_time"] = data["acq_time"].apply(lambda x: f"{int(x):04d}" if pd.notna(x) else np.nan)
data["acq_time"] = pd.to_numeric(data["acq_time"], errors="coerce")

data = data.drop(columns=["instrument","acq_date"], errors="ignore")


# =====================================================
# CLEAN TARGET
# =====================================================

data = data.dropna(subset=["type"])

data["type"] = pd.to_numeric(data["type"], errors="coerce")

data = data.dropna(subset=["type"])

data["type"] = data["type"].astype(int)


# =====================================================
# BASIC CLEANING
# =====================================================

# duplicate
duplicate_mask = data.duplicated()

for idx in data[duplicate_mask].index:
    cleaning_log[idx] = "duplicate"

data = data.drop_duplicates()


# koordinat indonesia
geo_mask = ~(
    (data["latitude"].between(-11,6)) &
    (data["longitude"].between(95,141))
)

for idx in data[geo_mask].index:
    cleaning_log[idx] = "outside_indonesia"

data = data[
    (data["latitude"].between(-11,6)) &
    (data["longitude"].between(95,141))
]


# confidence
if "confidence" in data.columns:

    conf_mask = ~data["confidence"].between(0,100)

    for idx in data[conf_mask].index:
        cleaning_log[idx] = "invalid_confidence"

    data = data[data["confidence"].between(0,100)]


# frp
if "frp" in data.columns:

    frp_mask = data["frp"] < 0

    for idx in data[frp_mask].index:
        cleaning_log[idx] = "negative_frp"

    data = data[data["frp"] >= 0]


# brightness
bright_mask = data["brightness"] >= 500

for idx in data[bright_mask].index:
    cleaning_log[idx] = "brightness_outlier"

data = data[data["brightness"] < 500]


# =====================================================
# OUTLIER REMOVAL (IQR)
# =====================================================

def remove_outlier_iqr(df,col):

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (df[col] < lower) | (df[col] > upper)

    for idx in df[mask].index:
        cleaning_log[idx] = f"{col}_iqr_outlier"

    return df[(df[col] >= lower) & (df[col] <= upper)]


for col in ["brightness","bright_t31","frp"]:

    if col in data.columns:

        data = remove_outlier_iqr(data,col)

print("Final Shape :", data.shape)


# =====================================================
# DATA YANG TERHAPUS SAAT CLEANING
# =====================================================

removed_data = data_before_cleaning.loc[
    ~data_before_cleaning.index.isin(data.index)
].copy()

removed_data["cleaning_reason"] = removed_data.index.map(cleaning_log)

print("\n===== DATA REMOVED DURING CLEANING =====")
print("Jumlah data terhapus :", removed_data.shape[0])

print("\n===== CLEANING SUMMARY =====")
print(removed_data["cleaning_reason"].value_counts())

print("\nSample removed data:")
print(removed_data.head(10))

removed_data.to_csv(f"{OUTPUT_DIR}/removed_data_cleaning.csv", index=False)


# =====================================================
# 1 BAR DISTRIBUSI KELAS HOTSPOT
# =====================================================

plt.figure(figsize=(8,6))

class_counts = data["type"].value_counts().sort_index()

class_counts.plot(kind="bar")

plt.title("Distribusi Kelas Hotspot (Type)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")

for i,v in enumerate(class_counts):
    plt.text(i,v,str(v),ha="center")

savefig("01_bar_distribution_type.png")


# =====================================================
# DATA SPLITTING
# =====================================================

X = data.drop(columns=["type"])
y = data["type"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)


# =====================================================
# 2 STRATIFIED SPLIT DISTRIBUTION
# =====================================================

original = y.value_counts(normalize=True).sort_index()
train = y_train.value_counts(normalize=True).sort_index()
test = y_test.value_counts(normalize=True).sort_index()

df_split = pd.DataFrame({

    "Original":original,
    "Training":train,
    "Testing":test

})

plt.figure(figsize=(10,6))

df_split.plot(kind="bar")

plt.title("Stratified Train Test Split Distribution")
plt.xlabel("Class")
plt.ylabel("Proportion")

savefig("02_stratified_split_distribution.png")


# =====================================================
# PREPROCESS PIPELINE
# =====================================================

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if X[c].dtype!="object"]

preprocessor = ColumnTransformer([

    ("num",MinMaxScaler(),num_cols),
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)

])


# =========================================================
# MODEL TRAINING
# =========================================================

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    "model__n_estimators": [300],
    "model__max_depth": [None],
    "model__min_samples_split": [2],
    "model__min_samples_leaf": [1],
    "model__max_features": ["sqrt"],
    "model__ccp_alpha": [0.0]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=10000))
])

lr_param_grid = {
    "model__C": [1.0],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
    "model__max_iter": [10000]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_


# =====================================================
# SIMPAN MODEL TERBAIK
# =====================================================

models = {
    "Random Forest": best_rf,
    "Logistic Regression": best_lr
}


# =====================================================
# TRAIN VS TEST ACCURACY
# =====================================================

train_acc=[]
test_acc=[]
names=[]

for name,model in models.items():

    train_pred=model.predict(X_train)
    test_pred=model.predict(X_test)

    train_acc.append(accuracy_score(y_train,train_pred))
    test_acc.append(accuracy_score(y_test,test_pred))

    names.append(name)

plt.figure(figsize=(8,6))

plt.plot(names,train_acc,marker="o",label="Training Accuracy")
plt.plot(names,test_acc,marker="o",label="Testing Accuracy")

plt.title("Perbandingan Accuracy Training vs Testing")

plt.ylabel("Accuracy")

plt.legend()

savefig("03_train_vs_test_accuracy.png")


print("FINISH")