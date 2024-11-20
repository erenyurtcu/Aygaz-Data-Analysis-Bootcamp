# Aygaz Veri Analisti Bootcamp Projesi

## 📋 Proje Açıklaması
Bu proje, Aygaz Veri Analisti Bootcamp kapsamında geliştirilmiş bir çalışmadır. Proje, veri analizi ve görselleştirme yoluyla anlamlı içgörüler elde etmeyi hedeflemektedir. Python ile veri ön işleme, keşifsel veri analizi (EDA) ve görselleştirme adımları gerçekleştirilmiştir.

GitHub ve Kaggle projelerine aşağıdaki bağlantılardan ulaşabilirsiniz:
- [GitHub: Aygaz Data Analyst Bootcamp](https://github.com/erenyurtcu/Aygaz-Data-Analysis-Bootcamp)
- [Kaggle: Aygaz Data Analyst Bootcamp](https://www.kaggle.com/code/erenyurtcu/aygaz-data-analyst-bootcamp)

## 📂 Dosya Yapısı
- `aygaz_data_analyst_bootcamp.py`: Tüm analizlerin ve görselleştirmelerin yer aldığı Python dosyası.

## 🛠️ Kullanılan Teknolojiler
- Python
- Pandas
- Matplotlib / Seaborn
- NumPy

## ⚙️ Gereksinimler
Projeyi çalıştırmak için aşağıdaki bileşenlerin yüklü olması gerekir:
- Python 3.7 veya üzeri
- Gerekli Python kütüphaneleri:
  ```bash
  pip install pandas numpy matplotlib seaborn
  ```

## 🚀 Nasıl Çalıştırılır?
1. Bu depoyu klonlayın:
   ```bash
   git clone https://github.com/erenyurtcu/Aygaz-Data-Analysis-Bootcamp
   ```
2. `aygaz_data_analyst_bootcamp.py` dosyasını çalıştırın:
   ```bash
   python aygaz_data_analyst_bootcamp.py
   ```

## 📈 Proje Hedefleri
- Verilerin temizlenmesi ve ön işlenmesi.
- Keşifsel veri analizi (EDA) ile anlamlı içgörüler elde etmek.
- Analiz sonuçlarını görselleştirme.

## 📊 Proje Uygulamaları ve Algoritma Seçimi

### Projenin Kullanılabileceği Sektörler ve Amaçlar:

#### E-ticaret Sektörü:
- **Amaç**: Müşteri davranışlarını analiz etmek, satış trendlerini belirlemek ve stok yönetimini optimize etmek.
- Bu tür analizler, müşteri segmentasyonu, çapraz satış stratejileri, fiyatlandırma optimizasyonu ve envanter yönetiminde kullanılabilir.

#### Lojistik ve Tedarik Zinciri:
- **Amaç**: Talep tahmini yaparak ürünlerin doğru miktarlarda ve doğru zamanlarda stoklanmasını sağlamak.
- Lojistik süreçlerin optimizasyonu, tedarik zincirindeki darboğazları önlemek için bu analizlerden faydalanılabilir.

#### Perakende Analitiği:
- **Amaç**: Mağaza ve ürün performanslarını değerlendirmek, satış artırıcı kampanyalar düzenlemek.
- Özellikle popüler ürünlerin belirlenmesi ve az satan ürünlerin stratejik olarak yeniden konumlandırılması için önemlidir.

#### Bölgesel Pazarlama:
- **Amaç**: Satışların coğrafi analizini yaparak bölgesel farklılıklara uygun pazarlama stratejileri geliştirmek.

### Algoritma Seçimi ve Gerekçeleri:

#### K-Means Clustering:
- **Neden Seçilmeli**: Müşteri segmentasyonu ve ürün kümelenmesi için basit ve etkili bir algoritmadır. Müşterilerin veya ürünlerin benzerliklerine göre gruplandırılması sağlar.

#### Apriori Algoritması:
- **Neden Seçilmeli**: Ürün ilişkilendirme ve çapraz satış stratejileri için kullanılabilir. Özellikle hangi ürünlerin birlikte satın alındığını analiz etmek için uygundur.

#### Time Series Algorithms (ARIMA, LSTM):
- **Neden Seçilmeli**: Talep tahmini ve satış trendlerini belirlemek için zaman serisi analizleri gerekir. ARIMA modeli basit senaryolar için, LSTM ise daha karmaşık ve uzun dönemli tahminler için kullanılabilir.

#### Anomaly Detection Algorithms (Isolation Forest, DBSCAN):
- **Neden Seçilmeli**: Aykırı değerleri tespit etmek ve anormal müşteri davranışlarını belirlemek için bu algoritmalar uygundur.

#### Regression Modelleri:
- **Neden Seçilmeli**: Satış fiyatlarının ürün özelliklerine göre tahmini için kullanılabilir. Özellikle ürün fiyatlandırma optimizasyonu için uygundur.

Bu algoritmalar, proje hedeflerine göre seçilerek uygulanabilir ve sektör bazlı çözümlerin verimliliğini artırabilir.

## 📝 Notlar
- Projede kullanılan veri seti ve diğer detaylar GitHub ve Kaggle projelerinde açıklanmıştır.
- Herhangi bir sorunla karşılaşırsanız lütfen GitHub veya Kaggle üzerinden iletişime geçin.
