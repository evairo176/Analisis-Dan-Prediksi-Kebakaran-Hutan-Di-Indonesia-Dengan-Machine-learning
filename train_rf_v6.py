# ============================================
# MODIS FIRMS INDONESIA - FINAL PIPELINE (COMPLETE & VALID)
# - No Data Leakage
# - Error Detection
# - Advanced Dataset Repair (IQR)
# - Compare Models: RF, DT, KNN, NB, LR
# - ROC/AUC Multiclass per Model + Combined (micro)
# - Extra Plots
# ============================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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
# CLEAN TARGET
# =========================
data = data.dropna(subset=["type"])
data["type"] = pd.to_numeric(data["type"], errors="coerce")
data = data.dropna(subset=["type"])
data["type"] = data["type"].astype(int)


# =========================
# BASIC CLEANING
# =========================
data = data.drop_duplicates()

data = data[
    (data["latitude"].between(-11, 6)) &
    (data["longitude"].between(95, 141))
]

if "confidence" in data.columns:
    data = data[data["confidence"].between(0,100)]

if "frp" in data.columns:
    data = data[data["frp"] >= 0]

data = data[data["brightness"] < 500]


# =========================
# IQR OUTLIER REMOVAL
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

print("Final shape:", data.shape)

# =========================
# VISUALISASI TAMBAHAN
# =========================

# 1️⃣ Bar Distribution
plt.figure(figsize=(8,6))
class_counts = data["type"].value_counts().sort_index()
class_counts.plot(kind="bar")
plt.title("Distribusi Kelas Hotspot (Type)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")
for i, v in enumerate(class_counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
savefig("02_bar_distribution_type.png")

# 2️⃣ Pie Distribution
plt.figure(figsize=(6,6))
class_counts.plot(kind="pie", autopct='%1.1f%%')
plt.ylabel("")
plt.title("Pie Distribution Hotspot Type")
savefig("02_pie_distribution_type.png")

# 3️⃣ Location Plot
plt.figure(figsize=(20,10))
plt.scatter(data["longitude"], data["latitude"],
            c=data["brightness"], s=2)
plt.colorbar(label="Brightness")
plt.title("Location Plot Hotspot Indonesia")
savefig("02_location_plot.png")

# 4️⃣ Trend per Year
counts = data["Year"].value_counts().sort_index()
plt.figure(figsize=(16,5))
plt.plot(counts.index, counts.values, marker="o")
plt.title("Trend Hotspot per Year")
savefig("03_trend_per_year.png")

# 5️⃣ Monthly Mean Brightness
df_month = data.groupby(["type","Month"])["brightness"].mean().reset_index()
plt.figure(figsize=(16,8))
for t in sorted(df_month["type"].unique()):
    tmp = df_month[df_month["type"]==t]
    plt.plot(tmp["Month"], tmp["brightness"], marker="o", label=f"type {t}")
plt.legend()
plt.title("Monthly Mean Brightness per Type")
savefig("04_monthly_brightness.png")

# 6️⃣ Correlation Heatmap
num_cols_corr = data.select_dtypes(include=[np.number]).columns.tolist()
num_cols_corr.remove("type")
plt.figure(figsize=(16,6))
sns.heatmap(data[num_cols_corr].corr(), vmin=-1, vmax=1, annot=True)
plt.title("Correlation Heatmap")
savefig("05_corr_heatmap.png")


# =========================
# MODEL PREP
# =========================
X = data.drop(columns=["type"])
y = data["type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# VISUALISASI STRATIFIED SPLIT
# =========================
plt.figure(figsize=(10,6))

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
plt.ylabel("Proporsi")
plt.xticks(rotation=0)

savefig("07_stratified_split_distribution.png")

classes = np.sort(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

cat_cols = [c for c in X.columns if X[c].dtype=="object"]
num_cols = [c for c in X.columns if X[c].dtype!="object"]

preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


# =========================================================
# RANDOM FOREST 
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

models = {
    "Random Forest": best_rf,
    "Logistic Regression": best_lr
}


# =========================
# CROSS VALIDATION
# =========================
print("\n=== 5-Fold CV ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")


# =========================
# EVALUATION + ROC AUC
# =========================

plt.figure(figsize=(9,7))
plt.plot([0,1],[0,1],"--", label="Random Guess")

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig(f"cm_{name.replace(' ','_')}.png")

    # =========================
    # ROC AUC MULTICLASS
    # =========================
    if hasattr(model.named_steps["model"], "predict_proba"):

        y_score = model.predict_proba(X_test)

        # Flatten multiclass
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())

        roc_auc = auc(fpr, tpr)

        print(f"AUC Score ({name}): {roc_auc:.4f}")

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

plt.legend()
plt.title("ROC Curve Comparison (Micro-Average)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

savefig("06_roc_compare.png")

# =========================================================
# FEATURE IMPORTANCE COMPARISON (1 GRAPH)
# =========================================================

# --- RANDOM FOREST ---
rf_model = best_rf.named_steps["model"]

feature_names = (
    num_cols +
    list(best_rf.named_steps["prep"]
         .named_transformers_["cat"]
         .get_feature_names_out(cat_cols))
)

importances_rf = rf_model.feature_importances_

# --- LOGISTIC REGRESSION ---
lr_model = best_lr.named_steps["model"]
coef_lr = np.mean(np.abs(lr_model.coef_), axis=0)

# Normalisasi supaya skala sebanding
importances_rf_norm = importances_rf / np.max(importances_rf)
coef_lr_norm = coef_lr / np.max(coef_lr)

# Buat dataframe gabungan
feat_compare = pd.DataFrame({
    "Feature": feature_names,
    "Random Forest": importances_rf_norm,
    "Logistic Regression": coef_lr_norm
})

# Ambil Top 10 berdasarkan rata-rata
feat_compare["Mean"] = feat_compare[["Random Forest", "Logistic Regression"]].mean(axis=1)
feat_compare = feat_compare.sort_values("Mean", ascending=False).head(10)

# Plot dalam 1 grafik
plt.figure(figsize=(12,8))

x = np.arange(len(feat_compare))
width = 0.35

plt.barh(x - width/2, feat_compare["Random Forest"], height=width, label="Random Forest")
plt.barh(x + width/2, feat_compare["Logistic Regression"], height=width, label="Logistic Regression")

plt.yticks(x, feat_compare["Feature"])
plt.xlabel("Normalized Importance")
plt.title("Feature Importance Comparison\nRandom Forest vs Logistic Regression")
plt.legend()

plt.gca().invert_yaxis()

savefig("09_feature_importance_comparison.png")

print("\nFinish code :).")