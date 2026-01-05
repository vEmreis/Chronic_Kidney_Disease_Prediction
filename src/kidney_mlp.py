import pandas as pd
import numpy as np
import joblib
import os


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# 1. VERIYI YUKLE

DATA_PATH = r"C:\Python\Chronic_Kidney_Disease_Prediction\data\kidney_disease.csv"


df = pd.read_csv(DATA_PATH)
print("Veri yüklendi.\n")
print(df.head())
print(df.shape)



# 2. ON ISLEME


# '?' olanları NaN yap
df.replace("?", np.nan, inplace=True)

# classification BOS OLAN SATIRLARI SIL (KRITIK!)
df.dropna(subset=["classification"], inplace=True)

# Sayisal kolonlari belirle
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Kategorik kolonlari belirle
categorical_cols = df.select_dtypes(include=["object"]).columns
categorical_cols = categorical_cols.drop("classification")

# Sayisal kolonlarda NaN → median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Kategorik kolonlarda NaN → mode
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Label Encoding
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# classification encode (ckd=1, notckd=0)
# classification temizleme (bosluk, tab vs.)
df["classification"] = df["classification"].astype(str).str.strip()

# encode
df["classification"] = df["classification"].map({
    "ckd": 1,
    "notckd": 0
})

# hala NaN varsa SIL (son emniyet)
df.dropna(subset=["classification"], inplace=True)


# SON KONTROL (NaN KALDI MI?)
print("\nNaN kontrolü:")
print(df.isnull().sum().sum())

# Temizlenmis veriyi kaydet
df.to_csv("../data/cleaned_kidney_data.csv", index=False)
print("\nTemizlenmiş veri kaydedildi.")
df.drop("id", axis=1, inplace=True)



# 3. OZELLIK - ETIKET AYIR

X = df.drop("classification", axis=1)
y = df["classification"]



# 4. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



# 5. STANDARDIZATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 6. MLP MODEL

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)



# 7. DEGERLENDIRME

y_pred = mlp.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import os

# klasor garanti
os.makedirs("../reports", exist_ok=True)


# CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix - MLP (CKD)")

plt.savefig("../reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# ACCURACY BAR PLOT

accuracy = accuracy_score(y_test, y_pred)

plt.figure()
plt.bar(["MLP Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("Model Accuracy")

plt.savefig("../reports/accuracy.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nPlotlar reports klasörüne kaydedildi.")

import joblib

joblib.dump(mlp, "../reports/mlp_model.pkl")
joblib.dump(scaler, "../reports/scaler.pkl")

print("Model ve scaler kaydedildi.")
