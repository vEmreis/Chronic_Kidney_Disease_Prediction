import pandas as pd
import joblib
import numpy as np

# Model ve scaler yolları
MODEL_PATH = "../reports/mlp_model.pkl"
SCALER_PATH = "../reports/scaler.pkl"

# Modeli ve scaler'ı yükle
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model yüklendi.")
print("Test edilecek örnek hasta verisi hazırlanıyor...\n")

# ÖRNEK HASTA VERİSİ
# (dataset'teki feature sırasına birebir uygun)
sample_data = {
    "age": 45,
    "bp": 80,
    "sg": 1.020,
    "al": 1,
    "su": 0,
    "rbc": 1,
    "pc": 1,
    "pcc": 0,
    "ba": 0,
    "bgr": 120,
    "bu": 36,
    "sc": 1.2,
    "sod": 138,
    "pot": 4.5,
    "hemo": 15,
    "pcv": 44,
    "wc": 7800,
    "rc": 5.1,
    "htn": 1,
    "dm": 0,
    "cad": 0,
    "appet": 1,
    "pe": 0,
    "ane": 0
}

# DataFrame'e çevir
X_sample = pd.DataFrame([sample_data])

# Ölçekleme
X_sample_scaled = scaler.transform(X_sample)

# Tahmin
prediction = model.predict(X_sample_scaled)[0]
probability = model.predict_proba(X_sample_scaled)[0]

# Sonuç
print("Tahmin Sonucu:")
if prediction == 1:
    print("👉 Kronik Böbrek Hastalığı (CKD) TESPİT EDİLDİ")
else:
    print("👉 Kronik Böbrek Hastalığı YOK")

print("\nOlasılıklar:")
print(f"CKD Yok: %{probability[0]*100:.2f}")
print(f"CKD Var: %{probability[1]*100:.2f}")
