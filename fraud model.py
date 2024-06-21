import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Dosya yolunu güncelleyin
file_path = '/Users/alikamiloglu/Downloads/fraud_data_set.xls - fraud.csv'  # Bu yolu kontrol edin ve dosya adının doğru olduğundan emin olun

# Dosya uzantısına dikkat ederek uygun okuma fonksiyonunu kullanın
# Eğer dosyanız CSV ise aşağıdaki satırı aktif hale getirin
# df = pd.read_csv(file_path)

# Eğer dosyanız Excel ise aşağıdaki satırı aktif hale getirin
df = pd.read_excel(file_path)  # Excel dosyası için

# İlk birkaç satırı kontrol edin
print("Veri setinin ilk birkaç satırı:\n", df.head())

# 1. Eksik Değerlerin Kontrolü ve Eksik Değerler Oluşturma
print("Eksik değerlerin sayısı:\n", df.isnull().sum())

# Eğitim amacıyla rastgele eksik değerler yaratıyoruz (Örnek olarak)
df.loc[0, 'amt'] = np.nan
df.loc[1, 'city'] = np.nan
df.loc[2, 'gender'] = np.nan

print("Eksik değerlerin tekrar kontrolü:\n", df.isnull().sum())

# Eksik Değerleri Doldurma
df['amt'].fillna(df['amt'].mean(), inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

print("Eksik değerlerin doldurulmasından sonraki durumu:\n", df.isnull().sum())

# 2. Gereksiz Değişkenlerin Çıkartılması
# Analiz için önemli olmayan bazı değişkenleri çıkarıyoruz
df = df.drop(columns=['first', 'last', 'street', 'state', 'job'])

# 3. Ölçeklendirme (Örneğin: 'amt' değişkenini)
scaler = StandardScaler()
df['amt_scaled'] = scaler.fit_transform(df[['amt']])

# 4. Nitel Değişkenlerin Kodlanması ve Gruplandırılması
# Kategorik değişkenlerin kodlanması
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_gender = encoder.fit_transform(df[['gender']])
df['gender_encoded'] = encoded_gender

# 5. Nicel Değişkenlerin Kategorize Edilmesi
# 'amt' değişkenini kategorilere ayırıyoruz
df['amt_category'] = pd.cut(df['amt'], bins=3, labels=['low', 'medium', 'high'])

print("Veri setinin işlenmiş hali:\n", df.head())

# 6. Verilerin Keşfedici Analizi

# Nitel değişkenlerin analizi
print("Gender Sıklık Tablosu:\n", df['gender'].value_counts())

# Nicel değişkenlerin analizi
print("İşlem Tutarı İstatistikleri:\n", df['amt'].describe())

# Dağılım Grafikleri
plt.hist(df['amt'], bins=20)
plt.title('İşlem Tutarı Dağılımı')
plt.xlabel('Tutar')
plt.ylabel('Frekans')
plt.show()

plt.boxplot(df['amt'].dropna())
plt.title('İşlem Tutarı Kutu Grafiği')
plt.ylabel('Tutar')
plt.show()

# Korelasyon Analizi
correlation_matrix = df.corr()
print("Korelasyon Matrisi:\n", correlation_matrix)

# 7. Modelleme ve Değerlendirme

# Hedef değişken
y = df['is_fraud']

# Özellikler (Bağımsız Değişkenler)
X = df.drop(columns=['is_fraud', 'trans_date_trans_time', 'city', 'dob', 'trans_num', 'amt', 'amt_category'])

# Eğitim ve test seti olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Modeli ile Eğitme
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans Değerlendirmesi
print("Model Performans Raporu:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# 8. Sonuçların Raporlanması

# Özellik önem dereceleri
importance = model.coef_[0]
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel('Özellik Önemi')
plt.ylabel('Özellikler')
plt.title('Özelliklerin Önemi')
plt.show()
