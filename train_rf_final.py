# ============================================
# MODIS FIRMS INDONESIA - FINAL PIPELINE
# ============================================

import os  # modul untuk operasi sistem seperti membuat folder
import warnings
warnings.filterwarnings("ignore")  # menonaktifkan warning agar output lebih bersih

import pandas as pd  # library manipulasi data
import numpy as np  # operasi numerik
import matplotlib.pyplot as plt  # visualisasi grafik
import seaborn as sns  # visualisasi statistik

from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV  # pembagian data dan cross validation
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, label_binarize  # preprocessing data
from sklearn.compose import ColumnTransformer  # transformer untuk pipeline preprocessing
from sklearn.pipeline import Pipeline  # pipeline preprocessing + model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc  # evaluasi model

from sklearn.linear_model import LogisticRegression  # algoritma Logistic Regression
from sklearn.ensemble import RandomForestClassifier  # algoritma Random Forest


# =====================================================
# OUTPUT FOLDER
# =====================================================
OUTPUT_DIR = "output"  # menentukan folder output
os.makedirs(OUTPUT_DIR, exist_ok=True)  # membuat folder jika belum ada

def savefig(name):  # fungsi untuk menyimpan grafik
    plt.savefig(f"{OUTPUT_DIR}/{name}", dpi=300, bbox_inches="tight")
    plt.close()


# =====================================================
# LOAD DATASET
# =====================================================
df_list = []

for year in range(2020,2025):  # membaca dataset hotspot dari tahun 2020–2024
    df_list.append(pd.read_csv(f"dataset/modis_{year}_Indonesia.csv"))

data = pd.concat(df_list, ignore_index=True)  # menggabungkan seluruh dataset

print("Initial Shape :", data.shape)  # menampilkan jumlah data awal

# Backup dataset sebelum cleaning
data_before_cleaning = data.copy()  # menyimpan dataset sebelum proses cleaning


# =====================================================
# FEATURE ENGINEERING
# =====================================================

data["acq_date"] = data["acq_date"].astype(str)  # memastikan kolom tanggal bertipe string

data[["Year","Month","Day"]] = data["acq_date"].str.split("-", expand=True)  # memecah tanggal menjadi tahun, bulan, hari

data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
data["Day"] = pd.to_numeric(data["Day"], errors="coerce")

data["acq_time"] = data["acq_time"].apply(lambda x: f"{int(x):04d}" if pd.notna(x) else np.nan)  # memformat waktu menjadi 4 digit
data["acq_time"] = pd.to_numeric(data["acq_time"], errors="coerce")

data = data.drop(columns=["instrument","acq_date"], errors="ignore")  # menghapus kolom yang tidak diperlukan


# =====================================================
# CLEAN TARGET
# =====================================================

data = data.dropna(subset=["type"])  # menghapus data tanpa label target

data["type"] = pd.to_numeric(data["type"], errors="coerce")  # konversi target menjadi numerik

data = data.dropna(subset=["type"])  # menghapus nilai target yang tidak valid

data["type"] = data["type"].astype(int)  # mengubah tipe target menjadi integer


# =====================================================
# BASIC CLEANING
# =====================================================

data = data.drop_duplicates()  # menghapus data duplikat

data = data[
    (data["latitude"].between(-11,6)) &
    (data["longitude"].between(95,141))
]  # memfilter hotspot yang berada di wilayah Indonesia

if "confidence" in data.columns:
    data = data[data["confidence"].between(0,100)]  # memastikan confidence valid

if "frp" in data.columns:
    data = data[data["frp"] >= 0]  # memastikan nilai frp tidak negatif

data = data[data["brightness"] < 500]  # menghapus brightness ekstrem


# =====================================================
# OUTLIER REMOVAL (IQR)
# =====================================================

def remove_outlier_iqr(df,col):  # fungsi untuk menghapus outlier menggunakan metode IQR

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[col] >= lower) & (df[col] <= upper)]


for col in ["brightness","bright_t31","frp"]:  # menghapus outlier pada fitur numerik penting

    if col in data.columns:

        data = remove_outlier_iqr(data,col)

print("Final Shape :", data.shape)  # menampilkan jumlah data setelah cleaning


# =====================================================
# IDENTIFIKASI DATA YANG TERHAPUS SAAT CLEANING
# =====================================================

removed_data = data_before_cleaning.loc[
    ~data_before_cleaning.index.isin(data.index)
].copy()  # mengambil data yang terhapus saat cleaning

def detect_cleaning_reason(row):  # menentukan alasan data dihapus

    if not (-11 <= row["latitude"] <= 6 and 95 <= row["longitude"] <= 141):
        return "outside_indonesia"

    if "confidence" in row.index:
        if pd.notna(row["confidence"]) and not (0 <= row["confidence"] <= 100):
            return "invalid_confidence"

    if "frp" in row.index:
        if pd.notna(row["frp"]) and row["frp"] < 0:
            return "negative_frp"

    if pd.notna(row["brightness"]) and row["brightness"] >= 500:
        return "brightness_outlier"

    return "iqr_outlier_or_duplicate"

removed_data["cleaning_reason"] = removed_data.apply(
    detect_cleaning_reason,
    axis=1
)

print("\n===== DATA REMOVED DURING CLEANING =====")
print("Jumlah data terhapus :", removed_data.shape[0])

print("\n===== CLEANING SUMMARY =====")
print(removed_data["cleaning_reason"].value_counts())

removed_data.to_csv(
    f"{OUTPUT_DIR}/removed_data_cleaning.csv",
    index=False
)  # menyimpan data yang terhapus


# =====================================================
# 1 BAR DISTRIBUSI KELAS HOTSPOT
# =====================================================

plt.figure(figsize=(8,6))

class_counts = data["type"].value_counts().sort_index()  # menghitung distribusi kelas

class_counts.plot(kind="bar")

plt.title("Distribusi Kelas Hotspot (Type)")
plt.xlabel("Kelas")
plt.ylabel("Jumlah Data")

for i,v in enumerate(class_counts):
    plt.text(i,v,str(v),ha="center")

savefig("01_bar_distribution_type.png")  # menyimpan grafik distribusi kelas


# =====================================================
# DATA SPLITTING
# =====================================================

X = data.drop(columns=["type"])  # fitur input
y = data["type"]  # target klasifikasi

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,  # 20% data testing
    random_state=42,
    stratify=y  # menjaga distribusi kelas
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

cat_cols = [c for c in X.columns if X[c].dtype=="object"]  # fitur kategorikal
num_cols = [c for c in X.columns if X[c].dtype!="object"]  # fitur numerik

preprocessor = ColumnTransformer([

    ("num",MinMaxScaler(),num_cols),  # normalisasi fitur numerik
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols)  # encoding kategori

])


# =========================================================
# 7. MODEL TRAINING
# Random Forest dan Logistic Regression
# =========================================================


# =========================================================
# RANDOM FOREST
# =========================================================

rf_pipeline = Pipeline([
    ("prep", preprocessor),  # preprocessing
    ("model", RandomForestClassifier(random_state=42))  # model Random Forest
])

# Hyperparameter Random Forest
rf_param_grid = {
    "model__n_estimators": [300],
    "model__max_depth": [None],
    "model__min_samples_split": [2],
    "model__min_samples_leaf": [1],
    "model__max_features": ["sqrt"],
    "model__ccp_alpha": [0.0]
}

# Grid Search Cross Validation
rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)  # training model Random Forest

best_rf = rf_grid.best_estimator_


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=10000))
])

# Hyperparameter Logistic Regression
lr_param_grid = {
    "model__C": [1.0],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
    "model__max_iter": [10000]
}

# Grid Search Cross Validation
lr_grid = GridSearchCV(
    lr_pipeline,
    lr_param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)  # training model Logistic Regression

best_lr = lr_grid.best_estimator_


# =========================================================
# SIMPAN MODEL TERBAIK
# =========================================================

models = {
    "Random Forest": best_rf,
    "Logistic Regression": best_lr
}


# =====================================================
# 3 TRAIN VS TEST ACCURACY
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


# =====================================================
# MODEL TESTING + CONFUSION MATRIX + ROC AUC
# =====================================================

plt.figure(figsize=(8,6))
plt.plot([0,1],[0,1],"--")

classes = np.sort(y.unique())
y_test_bin = label_binarize(y_test,classes=classes)

for name,model in models.items():

    print(f"\n=== {name} ===")

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test,y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    # Classification report
    print(classification_report(y_test,y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    savefig(f"cm_{name.replace(' ','_')}.png")

    # ROC AUC
    if hasattr(model.named_steps["model"],"predict_proba"):

        y_score = model.predict_proba(X_test)

        fpr,tpr,_ = roc_curve(y_test_bin.ravel(),y_score.ravel())

        roc_auc = auc(fpr,tpr)

        print(f"AUC Score ({name}): {roc_auc:.4f}")

        plt.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.3f})")

plt.legend()
plt.title("ROC Curve Comparison")

savefig("07_roc_compare.png")


# =====================================================
# 4 FEATURE IMPORTANCE COMPARISON
# =====================================================

rf = best_rf.named_steps["model"]
lr = best_lr.named_steps["model"]

feature_names = num_cols + list(
    best_rf.named_steps["prep"]
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols)
)

rf_imp = rf.feature_importances_
lr_imp = np.mean(np.abs(lr.coef_), axis=0)

# Normalisasi
rf_imp = rf_imp / rf_imp.max()
lr_imp = lr_imp / lr_imp.max()

df_feat = pd.DataFrame({
    "Feature": feature_names,
    "Random Forest": rf_imp,
    "Logistic Regression": lr_imp
})

df_feat["mean"] = df_feat[["Random Forest","Logistic Regression"]].mean(axis=1)

df_feat = df_feat.sort_values("mean",ascending=False).head(10)

plt.figure(figsize=(12,8))

x = np.arange(len(df_feat))
width = 0.35

plt.barh(x-width/2, df_feat["Random Forest"], width, label="Random Forest")
plt.barh(x+width/2, df_feat["Logistic Regression"], width, label="Logistic Regression")

plt.yticks(x, df_feat["Feature"])

plt.title("Feature Importance Comparison")

plt.xlabel("Normalized Importance")

plt.legend()

plt.gca().invert_yaxis()

savefig("04_feature_importance_comparison.png")


print("FINISH")