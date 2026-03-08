# ============================================
# MODIS FIRMS INDONESIA - FINAL PIPELINE (COMPLETE & VALID)
# - No Data Leakage
# - Error Detection
# - Advanced Dataset Repair (IQR)
# - Compare Models: RF, DT, KNN, NB, LR
# - ROC/AUC Multiclass per Model + Combined (micro)
# - Extra Plots: Location, Trend, KDE, Monthly Intensity, Heatmap
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# =========================
# OUTPUT FOLDER
# =========================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(filename):
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()

# =========================
# LOAD DATA
# =========================
df_list = []
for year in range(2020, 2025):
    df_list.append(pd.read_csv(f"dataset/modis_{year}_Indonesia.csv"))

dataset_raw = pd.concat(df_list, ignore_index=True)
data = dataset_raw.copy()

print("Initial shape:", data.shape)

# =========================
# MISSING CHECK
# =========================
print("\nMissing per column:\n", data.isna().sum())

plt.figure(figsize=(20,6))
mno.matrix(data)
plt.title("Missing Values Matrix")
savefig("01_missing_matrix.png")

# =========================
# FEATURE ENGINEERING
# =========================
data["acq_date"] = data["acq_date"].astype(str)
data[["Year","Month","Day"]] = data["acq_date"].str.split("-", expand=True)

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
data["Day"] = pd.to_numeric(data["Day"], errors="coerce")

data["acq_time"] = data["acq_time"].apply(lambda x: f"{int(x):04d}" if pd.notna(x) else np.nan)
data["acq_time"] = pd.to_numeric(data["acq_time"], errors="coerce")

data = data.drop(columns=["instrument","acq_date"], errors="ignore")

# =========================
# DROP TARGET MISSING
# =========================
data = data.dropna(subset=["type"])
data["type"] = pd.to_numeric(data["type"], errors="coerce")
data = data.dropna(subset=["type"])
data["type"] = data["type"].astype(int)

# =========================
# ERROR DETECTION + REPAIR
# =========================
print("\n=== ERROR DETECTION ===")
before = data.shape[0]

# Remove duplicates
data = data.drop_duplicates()

# Coordinate validation Indonesia
data = data[
    (data["latitude"].between(-11, 6)) &
    (data["longitude"].between(95, 141))
]

# Confidence valid
if "confidence" in data.columns:
    data = data[data["confidence"].between(0,100)]

# FRP non-negative
if "frp" in data.columns:
    data = data[data["frp"] >= 0]

# Brightness realistic
data = data[data["brightness"] < 500]

after = data.shape[0]
print("Rows removed basic cleaning:", before - after)

# =========================
# ADVANCED OUTLIER REMOVAL (IQR)
# =========================
def remove_outlier_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ["brightness","bright_t31","frp"]:
    if col in data.columns:
        data = remove_outlier_iqr(data, col)

print("Shape after IQR:", data.shape)

# =========================
# VISUALISASI
# =========================

# =========================
# DISTRIBUSI KELAS (BAR CHART)
# =========================
plt.figure(figsize=(8,6))

class_counts = data["type"].value_counts().sort_index()

class_counts.plot(kind="bar")

plt.title("Distribusi Kelas Hotspot (Type)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")

# Tambahkan label angka di atas batang
for i, v in enumerate(class_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')

savefig("02_bar_distribution_type.png")

# Location plot
plt.figure(figsize=(20,10))
plt.scatter(data["longitude"], data["latitude"],
            c=data["brightness"], s=2)
plt.colorbar(label="Brightness")
plt.title("Location Plot Hotspot Indonesia")
savefig("02_location_plot.png")

# Trend per year
counts = data["Year"].value_counts().sort_index()
plt.figure(figsize=(16,5))
plt.plot(counts.index, counts.values, marker="o")
plt.title("Trend Hotspot per Year")
savefig("03_trend_per_year.png")

# Monthly mean brightness
df_month = data.groupby(["type","Month"])["brightness"].mean().reset_index()
plt.figure(figsize=(16,8))
for t in sorted(df_month["type"].unique()):
    tmp = df_month[df_month["type"]==t]
    plt.plot(tmp["Month"], tmp["brightness"], marker="o", label=f"type {t}")
plt.legend()
plt.title("Monthly Mean Brightness per Type")
savefig("04_monthly_brightness.png")

# Correlation heatmap
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("type")
plt.figure(figsize=(16,6))
sns.heatmap(data[num_cols].corr(), vmin=-1, vmax=1, annot=True)
plt.title("Correlation Heatmap")
savefig("05_corr_heatmap.png")

# =========================
# MODEL PREP
# =========================
X = data.drop(columns=["type"])
y = data["type"]

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if X[c].dtype!="object"]

preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    # "Decision Tree": DecisionTreeClassifier(random_state=42),
    # "KNN": KNeighborsClassifier(n_neighbors=3),
    # "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=10000)
}

# models = {
#     "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
#     "Decision Tree": DecisionTreeClassifier(random_state=42),
#     "KNN": KNeighborsClassifier(n_neighbors=3),
#     "Naive Bayes": GaussianNB(),
#     "Logistic Regression": LogisticRegression(max_iter=10000)
# }

# =========================
# DATA SPLITTING SUMMARY TABLE
# =========================
print("\n=== HASIL DATA SPLITTING ===")

ratios = [0.1, 0.2, 0.3]
total_data = len(X)

print("\nTabel Hasil Data Splitting")
print("{:<10} {:<15} {:<15} {:<15}".format("Rasio", "Data Training", "Data Testing", "Total Data"))

for r in ratios:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=r,
        random_state=42,
        stratify=y
    )

    print("{:<10} {:<15} {:<15} {:<15}".format(
        f"{int((1-r)*100)}:{int(r*100)}",
        f"{len(X_tr):,}",
        f"{len(X_te):,}",
        f"{total_data:,}"
    ))

# =========================
# TRAIN TEST SPLIT
# =========================
print("\n=== DATA SPLITTING ===")
print("Total data:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set size:", len(X_train))
print("Testing set size :", len(X_test))

print("\nDistribusi kelas sebelum split:")
print(y.value_counts().sort_index())

print("\nDistribusi kelas training:")
print(y_train.value_counts().sort_index())

print("\nDistribusi kelas testing:")
print(y_test.value_counts().sort_index())

classes = np.sort(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

# =========================
# VISUALISASI STRATIFIED SPLIT
# =========================

plt.figure(figsize=(10,6))

# Hitung proporsi (%)
original_dist = y.value_counts(normalize=True).sort_index()
train_dist = y_train.value_counts(normalize=True).sort_index()
test_dist = y_test.value_counts(normalize=True).sort_index()

df_split = pd.DataFrame({
    "Original": original_dist,
    "Training (80%)": train_dist,
    "Testing (20%)": test_dist
})

df_split.plot(kind="bar")

plt.title("Perbandingan Distribusi Kelas\nOriginal vs Training vs Testing")
plt.xlabel("Kelas")
plt.ylabel("Proporsi (%)")
plt.xticks(rotation=0)

savefig("07_stratified_split_distribution.png")

# =========================
# CROSS VALIDATION
# =========================
print("\n=== 5-Fold CV ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# =========================
# EVALUATION + ROC
# =========================



plt.figure(figsize=(9,7))
plt.plot([0,1],[0,1],"--")

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # 🔥 CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig(f"cm_{name.replace(' ','_')}.png")

    # ROC
    if hasattr(pipe.named_steps["model"],"predict_proba"):
        y_score = pipe.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

plt.legend()
plt.title("ROC Curve Comparison (Micro Average)")
savefig("06_roc_compare.png")

# =========================
# TRAIN VS TEST ACCURACY PLOT
# =========================

train_acc_list = []
test_acc_list = []
model_names = []

for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    model_names.append(name)

plt.figure(figsize=(8,6))

plt.plot(model_names, train_acc_list, marker="o", label="Training Accuracy")
plt.plot(model_names, test_acc_list, marker="o", label="Testing Accuracy")

plt.title("Perbandingan Accuracy Training vs Testing")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.legend()

savefig("08_train_vs_test_accuracy.png")

print("\nFinish code :).")