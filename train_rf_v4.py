kenapa code nya jadi sedikit


# ============================================
# MODIS FIRMS INDONESIA - FINAL PIPELINE (VALID)
# - No Data Leakage
# - Compare Models: RF, DT, KNN, NB, LR
# - ROC/AUC Multiclass per Model + Combined (micro)
# - Extra Plots: Location, Trend, KDE, Monthly Intensity, Heatmap
#
# Output gambar otomatis ke folder: output/
# Overwrite otomatis jika file sudah ada
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

from sklearn.base import clone


# =========================
# 0) OUTPUT FOLDER
# =========================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(filename):
    """Simpan gambar ke output/ dengan overwrite otomatis"""
    plt.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300, bbox_inches="tight")
    plt.close()


# =========================
# 1) LOAD DATA
# =========================
df2024 = pd.read_csv("dataset/modis_2024_Indonesia.csv")
df2023 = pd.read_csv("dataset/modis_2023_Indonesia.csv")
df2022 = pd.read_csv("dataset/modis_2022_Indonesia.csv")
df2021 = pd.read_csv("dataset/modis_2021_Indonesia.csv")
df2020 = pd.read_csv("dataset/modis_2020_Indonesia.csv")

dataset_raw = pd.concat([df2020, df2021, df2022, df2023, df2024], ignore_index=True)

print("Dataset raw shape:", dataset_raw.shape)
print(dataset_raw.head())


# =========================
# 2) COPY DATA
# =========================
data = dataset_raw.copy()


# =========================
# 3) MISSING VALUES CHECK + VISUAL
# =========================
print("\nMissing values per column:")
print(data.isna().sum())

plt.figure(figsize=(20, 6))
mno.matrix(data)
plt.title("Missing Values Matrix (MODIS FIRMS)")
savefig("01_missing_matrix.png")


# =========================
# 4) FEATURE ENGINEERING: DATE
# =========================
# Pastikan acq_date string
data["acq_date"] = data["acq_date"].astype(str)

# Split YYYY-MM-DD -> Year Month Day
data[["Year", "Month", "Day"]] = data["acq_date"].str.split("-", expand=True)

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
data["Day"] = pd.to_numeric(data["Day"], errors="coerce")

# Rapikan acq_time (4 digit)
data["acq_time"] = data["acq_time"].apply(lambda x: f"{int(x):04d}" if pd.notna(x) else np.nan)
data["acq_time"] = pd.to_numeric(data["acq_time"], errors="coerce")


# =========================
# 5) DROP KOLOM YANG TIDAK DIPAKAI
# =========================
# instrument biasanya selalu MODIS
if "instrument" in data.columns:
    data = data.drop(columns=["instrument"], errors="ignore")

# acq_date sudah dipecah
data = data.drop(columns=["acq_date"], errors="ignore")


# =========================
# 6) DROP DATA YANG TARGETNYA KOSONG (PENTING!)
# =========================
# Ini yang benar untuk TA: label harus ground truth, bukan hasil imputasi
if "type" not in data.columns:
    raise ValueError("Kolom 'type' tidak ditemukan. Dataset harus punya kolom target 'type'.")

before = data.shape[0]
data = data.dropna(subset=["type"]).copy()
after = data.shape[0]

print(f"\nDrop missing target 'type': {before} -> {after}")


# =========================
# 7) PASTIKAN TYPE INTEGER
# =========================
data["type"] = pd.to_numeric(data["type"], errors="coerce")
data = data.dropna(subset=["type"])
data["type"] = data["type"].astype(int)


# =========================
# 8) VISUALISASI PIE CATEGORICAL (SEBELUM ENCODING)
# =========================
def pie_plot(df, col, filename, title):
    if col not in df.columns:
        return
    if df[col].dropna().shape[0] == 0:
        return

    plt.figure(figsize=(4, 4))
    df[col].dropna().value_counts().plot(kind="pie", autopct="%.2f")
    plt.title(title)
    plt.ylabel("")
    savefig(filename)


pie_plot(data, "type", "02_pie_type.png", "Type Distribution")
pie_plot(data, "satellite", "03_pie_satellite.png", "Satellite Distribution")
pie_plot(data, "daynight", "04_pie_daynight.png", "Day/Night Distribution")
pie_plot(data, "version", "05_pie_version.png", "Version Distribution")


# =========================
# 9) PLOT TAMBAHAN: LOCATION PLOT
# =========================
if all(col in data.columns for col in ["longitude", "latitude", "brightness"]):
    plt.figure(figsize=(20, 10))
    plt.scatter(
        data["longitude"],
        data["latitude"],
        c=data["brightness"],
        s=2
    )
    plt.title("Location Plot Hotspot (Brightness) - Indonesia 2020-2024")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Brightness")
    savefig("06_location_plot_hot_brightness.png")


# =========================
# 10) TREND HOTSPOT PER TAHUN
# =========================
if "Year" in data.columns:
    counts = data["Year"].value_counts().sort_index()

    plt.figure(figsize=(16, 5), dpi=120)
    plt.plot(counts.index, counts.values, marker="o")
    plt.title("Trend Hotspot per Tahun - Indonesia (2020-2024)")
    plt.xlabel("Year")
    plt.ylabel("Number of Hotspots")
    savefig("07_trend_hotspot_per_year.png")


# =========================
# 11) KDE DAY vs NIGHT (MASIH STRING)
# =========================
# daynight biasanya 'D' dan 'N'
if "daynight" in data.columns and "Year" in data.columns:
    plt.figure(figsize=(15, 5), dpi=120)

    if "D" in data["daynight"].unique():
        sns.kdeplot(data.loc[data["daynight"] == "D", "Year"], fill=True, label="Day", alpha=0.6)
    if "N" in data["daynight"].unique():
        sns.kdeplot(data.loc[data["daynight"] == "N", "Year"], fill=True, label="Night", alpha=0.6)

    plt.title("Density Hotspot berdasarkan Tahun (Day vs Night)")
    plt.xlabel("Year")
    plt.ylabel("Density")
    plt.legend()
    savefig("08_kde_day_vs_night.png")


# =========================
# 12) MONTHLY INTENSITY (MEAN)
# =========================
# Ini versi yang lebih benar: brightness rata-rata per bulan per type
if all(col in data.columns for col in ["type", "Month", "brightness"]):
    df_month = data.groupby(["type", "Month"])["brightness"].mean().reset_index()

    plt.figure(figsize=(16, 8), dpi=120)
    for t in sorted(df_month["type"].unique()):
        tmp = df_month[df_month["type"] == t]
        plt.plot(tmp["Month"], tmp["brightness"], marker="o", label=f"type {t}")

    plt.title("Monthly Mean Brightness per Type")
    plt.xlabel("Month")
    plt.ylabel("Mean Brightness")
    plt.legend()
    savefig("09_monthly_mean_brightness_by_type.png")


# =========================
# 13) CORRELATION HEATMAP (FITUR NUMERIC SAJA)
# =========================
numeric_cols_for_corr = data.select_dtypes(include=[np.number]).columns.tolist()
if "type" in numeric_cols_for_corr:
    numeric_cols_for_corr.remove("type")

plt.figure(figsize=(16, 6), dpi=120)
heatmap = sns.heatmap(
    data[numeric_cols_for_corr].corr(),
    vmin=-1,
    vmax=1,
    annot=True
)
heatmap.set_title("Correlation Heatmap (Features Only)", fontdict={"fontsize": 12}, pad=12)
savefig("10_corr_heatmap_features.png")


# =========================
# 14) MODELLING PREP
# =========================
X = data.drop(columns=["type"])
y = data["type"].astype(int)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

if X.shape[0] < 100:
    print("\nWARNING: Data sangat sedikit. Model bisa tidak stabil.")


# =========================
# 15) DEFINE FEATURE TYPES
# =========================
cat_cols = []
num_cols = []

for c in X.columns:
    if X[c].dtype == "object":
        cat_cols.append(c)
    else:
        num_cols.append(c)

print("\nCategorical cols:", cat_cols)
print("Numeric cols:", num_cols)


# =========================
# 16) PREPROCESSOR (NO LEAKAGE)
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)


# =========================
# 17) MODELS
# =========================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=10000, solver="lbfgs")
}


# =========================
# 18) TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

classes = np.sort(y.unique())
n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)


# =========================
# 19) CROSS VALIDATION (PIPELINE)
# =========================
print("\n=== 5-Fold Cross Validation Results (No Leakage) ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", clf)
    ])

    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    cv_results.append([name, scores.mean(), scores.std()])
    print(f"{name:18s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_df = pd.DataFrame(cv_results, columns=["Model", "Mean Accuracy", "Std"])
cv_df = cv_df.sort_values("Mean Accuracy", ascending=False)
cv_df.to_csv(f"{OUTPUT_DIR}/11_cv_results.csv", index=False)


# =========================
# 20) TRAIN + EVALUATE EACH MODEL (TEST SET)
# =========================
all_auc_micro = []
all_reports = []

for name, clf in models.items():
    print(f"\n=== MODEL: {name} ===")

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save report text
    report_txt = classification_report(y_test, y_pred)
    all_reports.append([name, acc, report_txt])

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    savefig(f"12_confusion_matrix_{name.replace(' ', '_').lower()}.png")

    # ROC/AUC micro (only if predict_proba exists)
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        y_score = pipe.predict_proba(X_test)

        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        all_auc_micro.append([name, auc_micro])
    else:
        all_auc_micro.append([name, np.nan])


# Save reports to CSV
report_df = pd.DataFrame(all_reports, columns=["Model", "Accuracy", "Classification_Report"])
report_df.to_csv(f"{OUTPUT_DIR}/12_model_reports.csv", index=False)


# =========================
# 21) ROC/AUC MULTICLASS PER MODEL (PLOT 1 PER MODEL)
# =========================
for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)

    if not hasattr(pipe.named_steps["model"], "predict_proba"):
        continue

    y_score = pipe.predict_proba(X_test)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average AUC = {roc_auc['micro']:.4f}")

    for i, c in enumerate(classes):
        plt.plot(fpr[i], tpr[i], label=f"class {c} AUC = {roc_auc[i]:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve (Multiclass) - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    savefig(f"13_roc_auc_multiclass_{name.replace(' ', '_').lower()}.png")


# =========================
# 22) ROC/AUC COMBINED ALL MODELS (MICRO)
# =========================
plt.figure(figsize=(9, 7))
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

auc_compare_rows = []

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)

    if not hasattr(pipe.named_steps["model"], "predict_proba"):
        print(f"Skip {name} karena tidak ada predict_proba()")
        continue

    y_score = pipe.predict_proba(X_test)

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    auc_compare_rows.append([name, auc_micro])
    plt.plot(fpr_micro, tpr_micro, label=f"{name} (AUC={auc_micro:.4f})")

plt.title("ROC Curve Micro-Average (Multiclass) - Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
savefig("14_roc_auc_compare_all_models.png")

auc_compare_df = pd.DataFrame(auc_compare_rows, columns=["Model", "Micro_AUC"])
auc_compare_df = auc_compare_df.sort_values("Micro_AUC", ascending=False)
auc_compare_df.to_csv(f"{OUTPUT_DIR}/14_auc_compare_all_models.csv", index=False)


print("\nSelesai! Semua output tersimpan di folder output/")
print("Cek file penting:")
print("- output/06_location_plot_hot_brightness.png")
print("- output/07_trend_hotspot_per_year.png")
print("- output/08_kde_day_vs_night.png")
print("- output/09_monthly_mean_brightness_by_type.png")
print("- output/10_corr_heatmap_features.png")
print("- output/14_roc_auc_compare_all_models.png")
print("- output/14_auc_compare_all_models.csv")

