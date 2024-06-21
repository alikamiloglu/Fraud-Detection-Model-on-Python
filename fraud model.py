# Gerekli Kütüphanelerin İçe Aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier

# Dosya Yolu
file_path = "/Users/alikamiloglu/Downloads/fraud_data_set.xls - fraud.csv"

# Veri Kümesinin Yüklenmesi
df = pd.read_csv(file_path)

# Veri Kümesinin İlk Satırlarının Gösterilmesi
print("Veri setinin ilk birkaç satırı:")
print(df.head())

# Eksik Değerlerin Kontrolü ve Sayılması
print("\nEksik değerlerin sayısı:")
print(df.isnull().sum())

# Eksik Değerlerin Doldurulması
df.fillna(method='ffill', inplace=True)

# Eksik Değerlerin Tekrar Kontrol Edilmesi
print("\nEksik değerlerin tekrar kontrolünü yapalım:")
print(df.isnull().sum())

# İşlenmiş Verinin İlk Satırlarının Gösterilmesi
print("\nVeri setinin işlenmiş hali:")
print(df.head())

# Cinsiyet (Gender) Sıklık Tablosu
print("\nGender (Cinsiyet) Sıklık Tablosu:")
print(df['gender'].value_counts())

# İşlem Tutarı İstatistikleri
print("\nİşlem Tutarı İstatistikleri:")
print(df['amt'].describe())

# İşlem Tutarı Histogramının Çizilmesi (0-750 Aralığı)
plt.figure(figsize=(10, 6))
plt.hist(df['amt'], bins=30, range=(0, 750), color='blue', edgecolor='black')
plt.title('İşlem Tutarı Histogramı (0-750)')
plt.xlabel('İşlem Tutarı ($)')
plt.ylabel('Frekans')
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/histogram.png")  # Histogramı Kaydetme
plt.show()

# Kategori Sıklık Tablosu
print("\nKategori Sıklık Tablosu:")
print(df['category'].value_counts())

# Kategori Frekans Dağılımının Görselleştirilmesi
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='category', order=df['category'].value_counts().index)
plt.title('Kategori Frekans Dağılımı')
plt.xlabel('Kategori')
plt.ylabel('Frekans')
plt.xticks(rotation=90)
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/category_distribution.png")  # Frekans Dağılımını Kaydetme
plt.show()

# Cinsiyet ve İşlem Tutarı Arasındaki İlişkinin İncelenmesi (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='gender', y='amt', data=df)
plt.title('Cinsiyete Göre İşlem Tutarı Dağılımı')
plt.xlabel('Cinsiyet')
plt.ylabel('İşlem Tutarı ($)')
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/gender_amt_boxplot.png")  # Cinsiyete Göre İşlem Tutarı Dağılımını Kaydetme
plt.show()

# Cinsiyet Dağılımı Pasta Grafiği
plt.figure(figsize=(8, 8))
df['gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
plt.title('Cinsiyet Dağılımı')
plt.ylabel('')
plt.savefig("/Users/alikamiloglu/Downloads/gender_distribution.png")  # Cinsiyet Dağılımını Kaydetme
plt.show()

# Kategorilere Göre Ortalama İşlem Tutarları
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='category', y='amt', estimator=np.mean, ci=None, order=df.groupby('category')['amt'].mean().sort_values(ascending=False).index)
plt.title('Kategorilere Göre Ortalama İşlem Tutarları')
plt.xlabel('Kategori')
plt.ylabel('Ortalama İşlem Tutarı ($)')
plt.xticks(rotation=90)
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/category_mean_amt.png")  # Kategorilere Göre Ortalama İşlem Tutarlarını Kaydetme
plt.show()

# Şehir Popülasyonu ve İşlem Tutarı Arasındaki İlişkinin İncelenmesi (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='city_pop', y='amt', data=df, alpha=0.5)
plt.title('Şehir Popülasyonu ve İşlem Tutarı Arasındaki İlişki')
plt.xlabel('Şehir Popülasyonu')
plt.ylabel('İşlem Tutarı ($)')
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/city_pop_amt_scatter.png")  # Şehir Popülasyonu ve İşlem Tutarı İlişkisini Kaydetme
plt.show()

# Is Fraud Kolonu'na Göre İşlem Tutarı Dağılımı (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_fraud', y='amt', data=df)
plt.title('Fraud Olma Durumuna Göre İşlem Tutarı Dağılımı')
plt.xlabel('Fraud Durumu')
plt.ylabel('İşlem Tutarı ($)')
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/is_fraud_amt_boxplot.png")  # Fraud Durumuna Göre İşlem Tutarı Dağılımını Kaydetme
plt.show()

# Bağımlı ve Bağımsız Değişkenlerin İncelenmesi
# Kullanılacak bağımsız değişkenler belirleniyor
features = ['amt', 'gender', 'city_pop', 'category']  # Örnek olarak bazı sütunlar
X = df[features]
y = df['is_fraud']

# Nitel Değişkenlerin Sayısal Hale Getirilmesi (Encoding)
X = pd.get_dummies(X, drop_first=True)

# Veri Eğitim ve Test Setlerine Ayrılıyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Kurulumu ve Değerlendirme

# Lojistik Regresyon Modeli
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLojistik Regresyon Sonuçları:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))
print("\nROC AUC Skoru:")
print(roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Naive Bayes Modeli
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print("\nNaive Bayes Sonuçları:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))
print("\nROC AUC Skoru:")
print(roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1]))

# Karar Ağacı Modeli
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nKarar Ağacı Sonuçları (max_depth=5):")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))
print("\nROC AUC Skoru:")
print(roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1]))

# Karar Ağacı Görselleştirilmesi
plt.figure(figsize=(20, 10))
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Not Fraud', 'Fraud'])
plt.title('Karar Ağacı Görselleştirmesi (max_depth=5)')
plt.savefig("/Users/alikamiloglu/Downloads/decision_tree.png")  # Karar Ağacı Görselleştirmesini Kaydetme
plt.show()

# Yapay Sinir Ağı (MLP) Modeli
nn_model = MLPClassifier(max_iter=1000)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

print("\nYapay Sinir Ağı Sonuçları:")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nn))
print("\nROC AUC Skoru:")
print(roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1]))

# Tüm Modeller İçin ROC Eğrisi Çizimi
plt.figure(figsize=(10, 6))

# Lojistik Regresyon ROC Eğrisi
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_log, tpr_log, label='Lojistik Regresyon (AUC = {:.2f})'.format(roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])))

# Naive Bayes ROC Eğrisi
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = {:.2f})'.format(roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1])))

# Karar Ağacı ROC Eğrisi
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_dt, tpr_dt, label='Karar Ağacı (AUC = {:.2f})'.format(roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])))

# Yapay Sinir Ağı ROC Eğrisi
fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_nn, tpr_nn, label='Yapay Sinir Ağı (AUC = {:.2f})'.format(roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])))

plt.title('Tüm Modeller İçin ROC Eğrisi')
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("/Users/alikamiloglu/Downloads/roc_curves.png")  # ROC Eğrisi Çizimini Kaydetme
plt.show()

# Tüm Model Sonuçlarının Karşılaştırılması
models = {
    'Lojistik Regresyon': log_model,
    'Naive Bayes': nb_model,
    'Karar Ağacı': dt_model,
    'Yapay Sinir Ağı': nn_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Modeli Sonuçları:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC AUC Skoru:")
    if hasattr(model, 'predict_proba'):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(roc_auc)
    else:
        print("Bu model için ROC AUC skoru hesaplanamadı.")

print("\nModel değerlendirme tamamlandı.")


