# ============================================================
# FIRE RISK MODEL - PONTIANAK (2021-2024)
# Random Forest + GEE Training Table + BMKG Harian
# FINAL FIX VERSION (sesuai kolom BMKG kamu)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import joblib


# ============================================================
# 1) PATH FILE
# ============================================================

GEE_TRAINING_CSV = "dataset/Pontianak_FireRisk_Training_Weekly_2021_2024 (1).csv"
BMKG_DAILY_CSV   = "dataset/pontianak_weather_daily_2021_2024.csv"


# ============================================================
# 2) LOAD DATA GEE
# ============================================================

gee = pd.read_csv(GEE_TRAINING_CSV)
gee.columns = [c.strip() for c in gee.columns]

print("=== Data GEE ===")
print("Shape:", gee.shape)
print("Columns:", gee.columns.tolist())
print(gee.head())


# ============================================================
# 2A) FIX KOLOM TANGGAL GEE (AUTO DETECT)
# ============================================================

possible_date_cols = ["week_start", "time", "label", "date", "week"]
date_col = None

for c in possible_date_cols:
    if c in gee.columns:
        date_col = c
        break

if date_col is None:
    raise ValueError(
        "Tidak ditemukan kolom tanggal di data GEE.\n"
        f"Kolom yang ada: {gee.columns.tolist()}"
    )

gee[date_col] = pd.to_datetime(gee[date_col], dayfirst=True, errors="coerce")
gee = gee.dropna(subset=[date_col]).copy()
gee = gee.rename(columns={date_col: "week_start"})


# ============================================================
# 3) LOAD DATA BMKG HARIAN
# ============================================================

bmkg = pd.read_csv(BMKG_DAILY_CSV)
bmkg.columns = [c.strip() for c in bmkg.columns]

if "date" not in bmkg.columns:
    raise ValueError(f"Kolom 'date' tidak ada di BMKG. Kolom tersedia: {bmkg.columns.tolist()}")

bmkg["date"] = pd.to_datetime(bmkg["date"], dayfirst=True, errors="coerce")
bmkg = bmkg.dropna(subset=["date"]).copy()

print("\n=== Data BMKG Harian (RAW) ===")
print("Shape:", bmkg.shape)
print("Columns:", bmkg.columns.tolist())
print(bmkg.head())


# ============================================================
# 4) FIX KOLOM BMKG (sesuai file kamu)
# ============================================================

# BMKG kamu punya kolom:
# TAVG = suhu rata-rata
# RH_AVG = kelembapan rata-rata
# RR = curah hujan

bmkg = bmkg.rename(columns={
    "TAVG": "tavg",
    "RH_AVG": "rh",
    "RR": "rain"
})

needed_bmkg_cols = ["date", "tavg", "rh", "rain"]
missing_bmkg = [c for c in needed_bmkg_cols if c not in bmkg.columns]

if len(missing_bmkg) > 0:
    raise ValueError(f"Kolom BMKG ini tidak ada di file kamu: {missing_bmkg}")

# pastikan numerik
bmkg["tavg"] = pd.to_numeric(bmkg["tavg"], errors="coerce")
bmkg["rh"]   = pd.to_numeric(bmkg["rh"], errors="coerce")
bmkg["rain"] = pd.to_numeric(bmkg["rain"], errors="coerce")

bmkg = bmkg.dropna(subset=["tavg", "rh", "rain"]).copy()


# ============================================================
# 5) HAPUS DUPLIKAT BMKG (karena ada tanggal double)
# ============================================================

# kalau ada 2 baris untuk tanggal yang sama, kita ambil rata-rata
bmkg_daily_clean = bmkg.groupby("date").agg({
    "tavg": "mean",
    "rh": "mean",
    "rain": "mean"
}).reset_index()

print("\n=== Data BMKG Harian (Clean, no duplicate date) ===")
print("Shape:", bmkg_daily_clean.shape)
print(bmkg_daily_clean.head())


# ============================================================
# 6) BMKG HARIAN -> MINGGUAN
# ============================================================

bmkg_daily_clean["week_start"] = bmkg_daily_clean["date"] - pd.to_timedelta(
    bmkg_daily_clean["date"].dt.dayofweek, unit="D"
)

weekly_bmkg = bmkg_daily_clean.groupby("week_start").agg({
    "tavg": "mean",
    "rh": "mean",
    "rain": "sum"
}).reset_index()

print("\n=== BMKG Mingguan ===")
print("Shape:", weekly_bmkg.shape)
print(weekly_bmkg.head())


# ============================================================
# 7) MERGE DATA GEE + BMKG
# ============================================================

df = gee.merge(weekly_bmkg, on="week_start", how="left")
df = df.dropna(subset=["tavg", "rh", "rain"]).copy()

print("\n=== Dataset Final (GEE + BMKG) ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


# ============================================================
# 8) PILIH FITUR + TARGET
# ============================================================

target = "fire"

features = [
    "NDVI", "LST", "RainSat",
    "elev", "slope", "lc",
    "tavg", "rh", "rain"
]

missing_features = [c for c in features + [target] if c not in df.columns]
if len(missing_features) > 0:
    raise ValueError(
        f"Kolom ini tidak ada di dataset final: {missing_features}\n"
        f"Kolom tersedia: {df.columns.tolist()}"
    )

# pastikan fitur numerik
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[target] = pd.to_numeric(df[target], errors="coerce")

df = df.dropna(subset=features + [target]).copy()

X = df[features].copy()
y = df[target].astype(int).copy()


# ============================================================
# 9) CEK IMBALANCE HOTSPOT
# ============================================================

print("\n=== Distribusi Label ===")
print("fire=1:", int((y == 1).sum()))
print("fire=0:", int((y == 0).sum()))
print("persen fire=1:", round((y == 1).mean() * 100, 4), "%")


# ============================================================
# 10) TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    # stratify=y
)


# ============================================================
# 11) TRAIN RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=10,
    min_samples_leaf=5,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)


# ============================================================
# 12) EVALUASI
# ============================================================

pred = rf.predict(X_test)
proba = rf.predict_proba(X_test)[:, 1]

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, pred, digits=4))

auc = roc_auc_score(y_test, proba)
print("\n=== AUC ===")
print(auc)


# ============================================================
# 13) FEATURE IMPORTANCE
# ============================================================

imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

print("\n=== Feature Importance ===")
print(imp)

imp.to_csv("feature_importance_pontianak.csv")


# ============================================================
# 14) SIMPAN MODEL
# ============================================================

joblib.dump(rf, "rf_fire_risk_pontianak.pkl")
print("\nModel tersimpan: rf_fire_risk_pontianak.pkl")


# ============================================================
# 15) SIMPAN OUTPUT PROBABILITAS RISIKO
# ============================================================

df["risk_prob"] = rf.predict_proba(X)[:, 1]

df_out = df[["week_start", "risk_prob"]].copy()
df_out.to_csv("pontianak_weekly_risk_probability.csv", index=False)

print("\nFile tersimpan: pontianak_weekly_risk_probability.csv")
print(df_out.head())


# ============================================================
# 16) SIMPAN DATASET FINAL
# ============================================================

df.to_csv("dataset_final_gee_bmkg.csv", index=False)
print("\nDataset final tersimpan: dataset_final_gee_bmkg.csv")
