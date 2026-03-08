# =====================================================
# FIRE RISK PREDICTION - RANDOM FOREST
# PONTIANAK (2021–2024)
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =====================================================
# 1. LOAD DATA
# =====================================================
bmkg = pd.read_csv('dataset/pontianak_weather_daily_2021_2024.csv')
gee  = pd.read_csv('dataset/Pontianak_VIIRS_NDVI_2021_2024.csv')

# =====================================================
# 2. PREPROCESS DATA BMKG (HARIAN → BULANAN)
# =====================================================
bmkg['date'] = pd.to_datetime(bmkg['date'], dayfirst=True)

bmkg['year'] = bmkg['date'].dt.year
bmkg['month'] = bmkg['date'].dt.month

bmkg_monthly = bmkg.groupby(['year','month']).agg({
    'TAVG': 'mean',
    'RH_AVG': 'mean',
    'RR': 'sum'
}).reset_index()

# =====================================================
# 3. GABUNGKAN DATA BMKG + GEE
# =====================================================
data = pd.merge(bmkg_monthly, gee, on=['year','month'], how='inner')

# =====================================================
# 4. LABEL RISIKO KEBAKARAN
# =====================================================
def label_fire(row):
    if row['TAVG'] > 29 and row['RH_AVG'] < 75 and row['RR'] == 0:
        return 'High'
    elif row['TAVG'] > 28 and row['RR'] < 5:
        return 'Moderate'
    else:
        return 'Low'

data['risk'] = data.apply(label_fire, axis=1)

# =====================================================
# 5. FITUR & TARGET
# =====================================================
X = data[['TAVG','RH_AVG','RR','hotspot_count','ndvi']]
y = data['risk']

# =====================================================
# 6. SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# 7. RANDOM FOREST MODEL
# =====================================================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# =====================================================
# 8. EVALUASI MODEL
# =====================================================
print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

# =====================================================
# 9. CONFUSION MATRIX (GRAFIK)
# =====================================================
cm = confusion_matrix(y_test, y_pred)
labels = rf.classes_

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix – Fire Risk Classification')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('Gambar_1_Confusion_Matrix.png', dpi=300)
plt.close()

# =====================================================
# 10. FEATURE IMPORTANCE
# =====================================================
fi = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(x='Importance', y='Feature', data=fi)
plt.title('Feature Importance – Random Forest')
plt.tight_layout()
plt.savefig('Gambar_2_Feature_Importance.png', dpi=300)
plt.close()

fi.to_csv('Feature_Importance.csv', index=False)

# =====================================================
# 11. DISTRIBUSI RISIKO KEBAKARAN
# =====================================================
plt.figure(figsize=(5,4))
sns.countplot(x=data['risk'], order=['Low','Moderate','High'])
plt.title('Distribusi Risiko Kebakaran')
plt.xlabel('Risk Level')
plt.ylabel('Jumlah Bulan')
plt.tight_layout()
plt.savefig('Gambar_3_Distribusi_Risiko.png', dpi=300)
plt.close()

# =====================================================
# 12. HOTSPOT PER TAHUN
# =====================================================
fires_per_year = data.groupby('year')['hotspot_count'].sum().reset_index()

plt.figure(figsize=(7,4))
plt.plot(fires_per_year['year'], fires_per_year['hotspot_count'], marker='o')
plt.title('Jumlah Hotspot Kebakaran per Tahun')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Hotspot')
plt.grid(True)
plt.tight_layout()
plt.savefig('Gambar_4_Hotspot_Tahunan.png', dpi=300)
plt.close()

# =====================================================
# 13. HOTSPOT PER BULAN
# =====================================================
fires_per_month = data.groupby('month')['hotspot_count'].sum().reset_index()

plt.figure(figsize=(8,4))
plt.bar(fires_per_month['month'], fires_per_month['hotspot_count'])
plt.title('Jumlah Hotspot Kebakaran per Bulan')
plt.xlabel('Bulan')
plt.ylabel('Jumlah Hotspot')
plt.xticks(range(1,13))
plt.tight_layout()
plt.savefig('Gambar_5_Hotspot_Bulanan.png', dpi=300)
plt.close()

# =====================================================
# 14. PETA RISIKO KEBAKARAN (HTML)
# =====================================================
dominant_risk = data['risk'].mode()[0]

color_map = {
    'Low': 'green',
    'Moderate': 'orange',
    'High': 'red'
}

pontianak_map = folium.Map(location=[-0.026, 109.333], zoom_start=10)

folium.Circle(
    location=[-0.026, 109.333],
    radius=25000,
    color=color_map[dominant_risk],
    fill=True,
    fill_color=color_map[dominant_risk],
    popup=f"Risiko Kebakaran Dominan: {dominant_risk}"
).add_to(pontianak_map)

pontianak_map.save('Peta_Risiko_Kebakaran_Pontianak.html')

print("\n=== PROSES SELESAI ===")
print("File grafik & peta berhasil dibuat.")