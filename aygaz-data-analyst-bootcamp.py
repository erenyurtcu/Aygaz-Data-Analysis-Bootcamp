#Veri Setini Yükleme

# Veri analizi için pandas'ı import et
import pandas as pd

# CSV dosyasından veri setini oku
data = pd.read_csv('online_retail.csv')

# İlk 5 satırı yazdır
print(data.head())

# Veri Setinin Null Değer Eklenmeden Önceki Hali
print(data.isnull().sum())

# Veri Setine Eksik Değerler Eklenmesi
# Gerçek dünya veri setlerinde eksik değerler sıkça karşılaşılan bir sorundur. Bu projede, veri temizleme ve işleme süreçlerini simüle etmek amacıyla veri setine rastgele %5 oranında eksik (NaN) değer eklenmiştir. Aşağıdaki fonksiyon, veri setine rastgele eksik değerler eklemek için kullanılmıştır.

import numpy as np
import random

def add_random_missing_values(dataframe: pd.DataFrame, missing_rate: float = 0.05) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults to 0.05.

    Returns:
        df_missing (pd.DataFrame): Processed DataFrame object.
    """
    # Get a copy of the dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number of missing values to add
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)

    # Generate random indices for rows and columns to assign NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing

data = add_random_missing_values(data, missing_rate=0.05)


# Veri Setinin Null Değer Eklendikten Sonraki Hali

print(data.isnull().sum())


# Veri Setinin İncelenmesi
# Aşağıdaki adımlarda, veri setinin genel yapısı incelenmiş ve eksiklik durumları analiz edilmiştir.

# Veri hakkında genel bilgi yazdırır
print("Data Info:")
print(data.info())

# Veri setindeki sütunların isimlerini yazdırır
print("\nData Columns:")
print(data.columns)

# Her bir sütundaki eksik değerlerin sayısını yazdırır
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

# Sayısal sütunların temel istatistiksel özetini yazdırır
print("\nStatistical Summary:")
print(data.describe())

# Veri setinin ilk 5 satırını yazdırır
print("\nFirst 5 Rows of the Data:")
print(data.head())

# Veri setinin son 5 satırını yazdırır
print("\nLast 5 Rows of the Data:")
print(data.tail())


# Veri Seti Analizi
# Aşağıdaki adımlarda veri seti üzerinde temel analizler gerçekleştirilmiştir.

# Fatura numaralarının benzersiz sayısını yazdırır
print("Number of Unique Invoice Numbers:")
print(data['InvoiceNo'].nunique())

# Sipariş verilen ülkelerin listesini yazdırır
print("\nUnique Countries:")
print(data['Country'].unique())

# Benzersiz müşteri sayısını yazdırır
print("\nNumber of Unique Customers:")
print(data['CustomerID'].nunique())

# Toplam sipariş sayısına göre en çok sipariş veren 5 müşteriyi yazdırır
print("\nTop 5 Customers by Total Orders:")
top_customers = data['CustomerID'].value_counts().head(5)
print(top_customers)

# Toplam sipariş sayısına göre en çok sipariş veren 5 ülkeyi yazdırır
print("\nTop 5 Countries by Total Orders:")
top_countries = data['Country'].value_counts().head(5)
print(top_countries)

# Satılan miktara göre en popüler 5 ürünü yazdırır
print("\nTop 5 Products by Quantity Sold:")
top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
print(top_products)

# Her bir sütundaki eksik değerlerin sayısını analiz eder ve yazdırır
print("\nMissing Values Analysis:")
missing_values = data.isnull().sum()
print(missing_values)

# Fatura bazında en yüksek toplam satışları yazdırır
print("\nInvoice-Based Total Sales (Top 5):")
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
top_invoices = data.groupby('InvoiceNo')['TotalPrice'].sum().sort_values(ascending=False).head(5)
print(top_invoices)

# Aylık toplam satışları özetler ve yazdırır
print("\nMonthly Sales Summary:")
data['Month'] = pd.to_datetime(data['InvoiceDate']).dt.month
monthly_sales = data.groupby('Month')['TotalPrice'].sum()
print(monthly_sales)


# Eksik Değerlerin Görselleştirilmesi: Heatmap
# Bu görselleştirme, hangi sütunlarda eksik değerlerin yoğun olduğunu ve eksikliklerin veri setindeki genel dağılımını göstermektedir.

# Veri setindeki eksik değerlerin dağılımını görselleştirmek için bir ısı haritası oluşturur
import seaborn as sns
import matplotlib.pyplot as plt

# Grafik boyutunu ayarlar
plt.figure(figsize=(12, 6))

# Eksik değerlerin bulunduğu alanları ısı haritasında gösterir
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')

# Grafik başlığını ayarlar
plt.title("Heatmap of Missing Values")

# Grafiği ekranda gösterir
plt.show()


# Veri Temizleme ve Ön İşleme

from sklearn.preprocessing import LabelEncoder

# 'Description' sütunundaki eksik değerler boş metin ('') ile doldurulur
data['Description'] = data['Description'].fillna('')

# 'CustomerID' sütununda eksik olan satırlar veri setinden çıkarılır
data = data.dropna(subset=['CustomerID'])

# 'InvoiceDate' sütunu datetime formatına dönüştürülür
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

# 'Quantity' ve 'UnitPrice' sütunlarında negatif ve sıfır değerler olan satırlar filtrelenir
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

# 'Country' sütunu sayısal değerlere dönüştürülerek yeni bir sütun ('Country_encoded') eklenir
data['Country_encoded'] = LabelEncoder().fit_transform(data['Country'])


# İstatistiksel Ölçümler: Standard Sapma, Medyan ve Mod

# 'Quantity' sütunundaki değerlerin standart sapmasını hesaplar ve yazdırır
print("Standard Deviation:")
print(data['Quantity'].std())

# 'Quantity' sütunundaki değerlerin medyanını hesaplar ve yazdırır
print("\nMedian:")
print(data['Quantity'].median())

# 'Quantity' sütunundaki en sık tekrar eden değeri (mod) hesaplar ve yazdırır
print("\nMode:")
print(data['Quantity'].mode()[0])


# Korelasyon Matrisi Görselleştirmesi

# 'Quantity', 'UnitPrice', 'TotalPrice' ve 'Country_encoded' sütunları arasındaki korelasyon matrisini hesaplar
correlation_matrix = data[['Quantity', 'UnitPrice', 'TotalPrice', 'Country_encoded']].corr()

# Korelasyon matrisini bir ısı haritası (heatmap) ile görselleştirir
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Grafiğin başlığını ayarlar
plt.title("Expanded Correlation Matrix")

# Grafiği ekranda gösterir
plt.show()


# En Çok Satılan 10 Ürün
# Veri setindeki ürünlerin toplam satış miktarlarına göre sıralanması yapılmış ve en çok satılan 10 ürün bir çubuk grafikle görselleştirilmiştir

# 'Description' sütununa göre ürünleri gruplandırır ve toplam 'Quantity' değerlerini hesaplar
# En çok satılan ilk 10 ürünü miktarlarına göre azalan sırada sıralar
top_products = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# İlk 10 ürünü çubuk grafik (bar plot) olarak görselleştirir
top_products.plot(kind='bar', figsize=(12, 6))

# Grafiğin başlığını, x ve y ekseni etiketlerini ayarlar
plt.title("Top 10 Most Sold Products")
plt.xlabel("Product")
plt.ylabel("Quantity Sold")

# X eksenindeki ürün isimlerini daha iyi görüntülemek için döndürür
plt.xticks(rotation=45)

# Grafiği ekranda gösterir
plt.show()


# En Yüksek Toplam Satışa Sahip 10 Ülke
# Veri setindeki ülkeler bazında toplam satışlar analiz edilmiş ve en yüksek toplam satışa sahip 10 ülke bir çubuk grafikle görselleştirilmiştir.

# 'Country' sütununa göre ülkeleri gruplandırır ve toplam 'TotalPrice' değerlerini hesaplar
# En yüksek toplam satışa sahip ilk 10 ülkeyi azalan sırada sıralar
country_sales = data.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

# İlk 10 ülkenin toplam satış değerlerini çubuk grafik (bar plot) olarak görselleştirir
country_sales.plot(kind='bar', figsize=(12, 6))

# Grafiğin başlığını, x ve y ekseni etiketlerini ayarlar
plt.title("Top 10 Countries by Total Sales")
plt.xlabel("Country")
plt.ylabel("Total Sales")

# X eksenindeki ülke isimlerini daha iyi görüntülemek için döndürür
plt.xticks(rotation=45)

# Grafiği ekranda gösterir
plt.show()

