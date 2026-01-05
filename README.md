Chronic Kidney Disease Prediction (MLP)

Bu proje, Kronik Böbrek Hastalığı (Chronic Kidney Disease – CKD) tahmini için makine öğrenmesi tabanlı bir sınıflandırma modeli geliştirmeyi amaçlamaktadır. Projede gerçek dünya verisi kullanılmış ve Çok Katmanlı Algılayıcı (Multilayer Perceptron – MLP) yapay sinir ağı modeli ile hastalık tahmini gerçekleştirilmiştir.

Proje, Veri Bilimine Giriş / Derin Öğrenme ve Uygulamaları dersi kapsamında hazırlanmıştır.

📌 Proje Özeti

Gerçek CKD veri seti kullanılmıştır

Veri ön işleme (eksik veri doldurma, encoding, ölçekleme) uygulanmıştır

MLP tabanlı yapay sinir ağı modeli eğitilmiştir

Model performansı accuracy, confusion matrix ve classification report ile değerlendirilmiştir

Eğitilen model ayrı bir test dosyası ile demo amaçlı test edilmiştir

📂 Proje Klasör Yapısı
Chronic_Kidney_Disease_Prediction/
│
├── data/
│   ├── kidney_disease.csv
│   └── cleaned_kidney_data.csv
│
├── src/
│   ├── kidney_mlp.py        # Model eğitimi
│   └── test_model.py        # Model test / demo
│
├── reports/
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   ├── mlp_model.pkl
│   └── scaler.pkl
│
├── notebooks/
├── references/
├── README.md

🧠 Kullanılan Yöntemler

Makine Öğrenmesi: Multilayer Perceptron (MLP)

Ön İşleme:

Eksik veri doldurma (median / mode)

Label Encoding

StandardScaler ile ölçekleme

Değerlendirme:

Accuracy

Confusion Matrix

Classification Report

▶️ Çalıştırma
1️⃣ Modeli Eğitmek

src klasörüne girerek:

python kidney_mlp.py


Bu adımda:

Model eğitilir

Değerlendirme metrikleri hesaplanır

Grafikler (reports/) klasörüne kaydedilir

Model ve scaler .pkl dosyaları olarak saklanır

2️⃣ Modeli Test Etmek (Demo)
python test_model.py


Bu adımda:

Kaydedilmiş model yüklenir

Örnek hasta verisi ile tahmin yapılır

CKD var / yok sonucu terminalde gösterilir

📊 Sonuçlar

Model, test veri seti üzerinde yüksek doğruluk oranı elde etmiştir. Confusion matrix sonuçları, modelin CKD ve CKD olmayan sınıfları başarılı bir şekilde ayırt edebildiğini göstermektedir. Elde edilen sonuçlar, MLP modelinin CKD tahmini için etkili bir yöntem olduğunu ortaya koymaktadır.

📚 Veri Seti

Kaynak: Kaggle – Chronic Kidney Disease Dataset

Veri seti 400 örnek ve 26 öznitelikten oluşmaktadır

🎓 Akademik Not

Bu proje eğitim amaçlıdır ve bir karar destek sistemi prototipi olarak geliştirilmiştir. Klinik kullanım için ek doğrulama ve uzman değerlendirmesi gereklidir.

👤 Hazırlayan

Emre Eriş
Bilgisayar Mühendisliği
Makine Öğrenmesi / Veri Bilimi Projesi