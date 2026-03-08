# ============================================
# MODIS FIRMS INDONESIA - CLEAN + MODELING
# + ROC/AUC MULTICLASS RANDOM FOREST
# + ROC/AUC GABUNGAN SEMUA MODEL (MICRO)
# + TAMBAHAN PLOT (LOKASI, TREND, KDE, MONTHLY INTENSITY, HEATMAP)
#
# Output gambar otomatis ke folder: output/
# Overwrite otomatis jika file sudah ada
# ============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import missingno as mno

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# =========================
# 0) OUTPUT FOLDER
# =========================
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(filename):
    """Helper: simpan gambar ke output/ dengan overwrite otomatis"""
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

dataset_raw = pd.concat([df2020,df2021,df2022,df2023, df2024], ignore_index=True)

print("Dataset raw shape:", dataset_raw.shape)
print(dataset_raw.head())


# =========================
# 2) COPY DATA
# =========================
data = dataset_raw.copy()


# =========================
# 3) CHECK MISSING VALUES
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
data["acq_date"] = data["acq_date"].astype(str)
data[["Year", "Month", "Day"]] = data["acq_date"].str.split("-", expand=True)

data["Year"] = data["Year"].astype(int)
data["Month"] = data["Month"].astype(int)
data["Day"] = data["Day"].astype(int)

# Rapikan waktu (jadi 4 digit)
data["acq_time"] = data["acq_time"].apply(lambda x: f"{int(x):04d}")
data["acq_time"] = data["acq_time"].astype(int)


# =========================
# 5) PIE CHART CATEGORICAL
# =========================
def pie_plot(col, filename, title):
    if col not in data.columns:
        return
    if data[col].dropna().shape[0] == 0:
        return

    plt.figure(figsize=(4, 4))
    data[col].dropna().value_counts().plot(kind="pie", autopct="%.2f")
    plt.title(title)
    plt.ylabel("")
    savefig(filename)


pie_plot("type", "02_pie_type.png", "Type Distribution")
pie_plot("satellite", "03_pie_satellite.png", "Satellite Distribution")
pie_plot("instrument", "04_pie_instrument.png", "Instrument Distribution")
pie_plot("daynight", "05_pie_daynight.png", "Day/Night Distribution")
pie_plot("version", "06_pie_version.png", "Version Distribution")


# =========================
# 6) DATA CLEANING
# =========================
# Drop instrument karena biasanya cuma MODIS semua
if "instrument" in data.columns:
    data = data.drop(["instrument"], axis=1)

# Drop acq_date karena sudah jadi Year Month Day
if "acq_date" in data.columns:
    data = data.drop(["acq_date"], axis=1)


# =========================
# 7) ENCODING
# =========================
cat_cols = ["satellite", "daynight", "version"]

for col in cat_cols:
    if col in data.columns:
        data[col] = data[col].astype(str)
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])


# =========================
# 8) NORMALIZATION (MINMAX)
# =========================
def minmax(df, col):
    if col in df.columns:
        mn = df[col].min()
        mx = df[col].max()
        if mx != mn:
            df[col] = (df[col] - mn) / (mx - mn)


for col in ["bright_t31", "brightness", "frp"]:
    minmax(data, col)


# =========================
# 9) IMPUTASI TYPE (JIKA ADA MISSING TYPE)
# =========================
if "type" not in data.columns:
    raise ValueError("Kolom 'type' tidak ada. Dataset kamu harus punya kolom type.")

train_data = data[data["type"].notna()].copy()
test_data = data[data["type"].isna()].copy()

print("\nTrain rows (type tersedia):", train_data.shape)
print("Test rows (type missing):", test_data.shape)

if test_data.shape[0] > 0:
    X_train_imp = train_data.drop(["type"], axis=1)
    y_train_imp = train_data["type"].astype(int)

    X_test_imp = test_data.drop(["type"], axis=1)

    # Naikkan max_iter supaya tidak ConvergenceWarning
    imputer_model = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=42)
    imputer_model.fit(X_train_imp, y_train_imp)

    y_pred_imp = imputer_model.predict(X_test_imp)
    test_data["type"] = y_pred_imp

    data = pd.concat([train_data, test_data], ignore_index=True)
    print("\nMissing type berhasil diisi dengan Logistic Regression.")
else:
    print("\nTidak ada missing value di kolom type. Skip imputasi.")


print("\nMissing values setelah imputasi:")
print(data.isna().sum())


# ==========================================================
# 10) TAMBAHAN PLOT: LOCATION PLOT (HOT COLORMAP)
# ==========================================================
# (Ini versi yang kamu minta: c=brightness colormap=hot)

if all(col in data.columns for col in ["longitude", "latitude", "brightness"]):
    plt.figure(figsize=(20, 10))
    plt.scatter(
        data["longitude"],
        data["latitude"],
        c=data["brightness"],
        s=2
    )
    plt.title("Location Plot Hotspot (Brightness) - Indonesia 2023-2024")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Brightness")
    savefig("07_location_plot_hot_brightness.png")


# ==========================================================
# 11) TAMBAHAN PLOT: TREND HOTSPOT PER TAHUN
# ==========================================================
def plottrend(df, title="", xlabel="Year", ylabel="Number of Hotspots", dpi=120):
    counts = df["Year"].value_counts().sort_index()

    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(counts.index, counts.values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    savefig("08_trend_hotspot_per_year.png")


if "Year" in data.columns:
    plottrend(
        data,
        title="Trend Hotspot per Tahun - Indonesia (2023-2024)"
    )


# ==========================================================
# 12) TAMBAHAN PLOT: KDE DAY vs NIGHT (BERDASARKAN YEAR)
# ==========================================================
# Catatan: daynight sudah di-encode (0/1).
# Ini akan tetap valid, tapi labelnya 0/1.
# Biasanya 0 = D, 1 = N (tergantung dataset)

if "daynight" in data.columns and "Year" in data.columns:
    plt.figure(figsize=(15, 5), dpi=120)

    sns.kdeplot(
        train_data.loc[train_data["daynight"] == 0, "Year"],
        fill=True,
        label="Day",
        alpha=0.6
    )
    sns.kdeplot(
        train_data.loc[train_data["daynight"] == 1, "Year"],
        fill=True,
        label="Night",
        alpha=0.6
    )

    plt.title("Density Hotspot berdasarkan Tahun (Day vs Night)")
    plt.xlabel("Year")
    plt.ylabel("Density")
    plt.legend()
    savefig("09_kde_day_vs_night.png")


# ==========================================================
# 13) TAMBAHAN PLOT: MONTHLY INTENSITY PER TYPE
# ==========================================================
if all(col in data.columns for col in ["type", "Month", "brightness"]):
    plt.figure(figsize=(16, 10), dpi=120)

    groups = data.groupby("type")
    for name, group in groups:
        plt.plot(
            group["Month"],
            group["brightness"],
            marker="o",
            linestyle="",
            label=f"type {name}"
        )

    plt.title("Forest Fire Intensity Monthly Trends (Brightness vs Month)")
    plt.xlabel("Month")
    plt.ylabel("Brightness (scaled)")
    plt.legend()
    savefig("10_monthly_intensity_by_type.png")


# ==========================================================
# 14) CORRELATION HEATMAP (ANNOT)
# ==========================================================
plt.figure(figsize=(16, 6), dpi=120)
heatmap = sns.heatmap(
    data.corr(numeric_only=True),
    vmin=-1,
    vmax=1,
    annot=True
)
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)
savefig("11_corr_heatmap_annot.png")


# =========================
# 15) MODELLING PREP
# =========================
X = data.drop(["type"], axis=1)
y = data["type"].astype(int)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

if X.shape[0] < 10:
    raise ValueError("Data terlalu sedikit untuk training model. Cek dataset kamu.")


# =========================
# 16) MODEL LIST
# =========================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=10000, solver="lbfgs")
}


# =========================
# 17) CROSS VALIDATION (MODEL COMPARISON)
# =========================
print("\n=== 5-Fold Cross Validation Results ===")
cv_results = []

for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    cv_results.append([name, scores.mean(), scores.std()])
    print(f"{name:18s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_df = pd.DataFrame(cv_results, columns=["Model", "Mean Accuracy", "Std"])
cv_df = cv_df.sort_values("Mean Accuracy", ascending=False)
cv_df.to_csv(f"{OUTPUT_DIR}/12_cv_results.csv", index=False)


# =========================
# 18) TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 19) FINAL MODEL: RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    random_state=42,
    n_estimators=300
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\n=== RANDOM FOREST FINAL RESULT ===")
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# =========================
# 20) CONFUSION MATRIX PLOT
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
savefig("13_confusion_matrix_rf.png")


# =========================
# 21) FEATURE IMPORTANCE (RF)
# =========================
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

feat_imp.to_csv(f"{OUTPUT_DIR}/14_feature_importance.csv", index=False)

plt.figure(figsize=(10, 5))
plt.bar(feat_imp["feature"], feat_imp["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Feature")
plt.ylabel("Importance")
savefig("14_feature_importance_rf.png")


# =========================
# 22) ROC / AUC (MULTICLASS) - RANDOM FOREST
# =========================
classes = np.sort(y.unique())
n_classes = len(classes)

y_test_bin = label_binarize(y_test, classes=classes)
y_score_rf = rf.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score_rf.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(8, 6))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average AUC = {roc_auc['micro']:.4f}"
)

for i, c in enumerate(classes):
    plt.plot(fpr[i], tpr[i], label=f"class {c} AUC = {roc_auc[i]:.4f}")

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve (Multiclass) - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

savefig("15_roc_auc_multiclass_rf.png")

auc_rows = []
for i, c in enumerate(classes):
    auc_rows.append([int(c), roc_auc[i]])
auc_rows.append(["micro", roc_auc["micro"]])

auc_df = pd.DataFrame(auc_rows, columns=["class", "auc"])
auc_df.to_csv(f"{OUTPUT_DIR}/15_auc_scores_rf.csv", index=False)


# =========================
# 23) ROC/AUC GABUNGAN SEMUA MODEL (MICRO-AVERAGE)
# =========================
plt.figure(figsize=(9, 7))

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")

auc_compare_rows = []

for name, clf in models.items():
    model_temp = clone(clf)
    model_temp.fit(X_train, y_train)

    if not hasattr(model_temp, "predict_proba"):
        print(f"Skip {name} karena tidak ada predict_proba()")
        continue

    y_score = model_temp.predict_proba(X_test)

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    auc_compare_rows.append([name, auc_micro])

    plt.plot(fpr_micro, tpr_micro, label=f"{name} (AUC={auc_micro:.4f})")

plt.title("ROC Curve Micro-Average (Multiclass) - Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")

savefig("16_roc_auc_compare_all_models.png")

auc_compare_df = pd.DataFrame(auc_compare_rows, columns=["Model", "Micro_AUC"])
auc_compare_df = auc_compare_df.sort_values("Micro_AUC", ascending=False)
auc_compare_df.to_csv(f"{OUTPUT_DIR}/16_auc_compare_all_models.csv", index=False)


print("\nSelesai! Semua output tersimpan di folder output/")
print("Cek file penting:")
print("- output/07_location_plot_hot_brightness.png")
print("- output/08_trend_hotspot_per_year.png")
print("- output/09_kde_day_vs_night.png")
print("- output/10_monthly_intensity_by_type.png")
print("- output/11_corr_heatmap_annot.png")
print("- output/15_roc_auc_multiclass_rf.png")
print("- output/16_roc_auc_compare_all_models.png")
print("- output/16_auc_compare_all_models.csv")
