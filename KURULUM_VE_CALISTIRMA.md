# Norm Kadro Analiz Sistemi - Kurulum ve Çalıştırma Rehberi

Bu doküman, Norm Kadro Analiz Sisteminin nasıl kurulacağını ve çalıştırılacağını adım adım açıklar.

## 📋 İçindekiler

1. [Sistem Gereksinimleri](#sistem-gereksinimleri)
2. [Proje Yapısı](#proje-yapısı)
3. [Kurulum Adımları](#kurulum-adımları)
4. [Çalıştırma Adımları](#çalıştırma-adımları)
5. [Sorun Giderme](#sorun-giderme)

---

## 🔧 Sistem Gereksinimleri

### Gerekli Yazılımlar

- **Python 3.8 veya üzeri** (Python 3.9+ önerilir)
- **pip** (Python paket yöneticisi)
- **Modern bir web tarayıcı** (Chrome, Firefox, Edge vb.)
- **İnternet bağlantısı** (CDN kaynakları için)

### Python Paketleri

Proje aşağıdaki Python paketlerini kullanır:
- FastAPI (Web framework)
- Uvicorn (ASGI sunucu)
- Pandas (Veri işleme)
- NumPy (Sayısal hesaplamalar)
- scikit-learn (Makine öğrenmesi)
- openpyxl (Excel dosyalarını okuma)

---

## 📁 Proje Yapısı

```
finalProje/
│
├── app.py                          # FastAPI backend sunucusu
├── personnel_analysis_ai.py         # AI analiz scripti
├── requirements.txt                # Python bağımlılıkları
│
├── index.html                      # Ana dashboard sayfası
├── app.js                          # Frontend JavaScript kodu
├── data.js                         # Statik veri dosyası
├── style.css                       # CSS stilleri
│
├── data/                           # Veri klasörü
│   ├── personel_durum.csv
│   ├── il_ilce_18yas_nufus.xlsx
│   ├── ilce_yuzolcum.csv
│   ├── norm_veteriner_hekim.csv
│   ├── norm_gida_mühendisi.csv
│   ├── norm_ziraat_mühendisi.csv
│   ├── tarim_alani.csv
│   ├── tarimsal_uretim.csv
│   ├── canli_hayvan.csv
│   └── denetim_sayisi.csv
│
├── personnel_analysis_results.csv  # Ana analiz sonuçları (oluşturulacak)
├── veteriner_analysis_results.csv  # Veteriner analiz sonuçları (oluşturulacak)
├── gida_analysis_results.csv       # Gıda mühendisi analiz sonuçları (oluşturulacak)
└── ziraat_analysis_results.csv     # Ziraat mühendisi analiz sonuçları (oluşturulacak)
```

---

## 🚀 Kurulum Adımları

### Adım 1: Python Kurulumunu Kontrol Edin

Terminal/PowerShell'de Python'un kurulu olup olmadığını kontrol edin:

```bash
python --version
```

veya

```bash
python3 --version
```

**Not:** Python 3.8 veya üzeri bir sürüm görmelisiniz. Eğer Python kurulu değilse, [python.org](https://www.python.org/downloads/) adresinden indirip kurun.

### Adım 2: Proje Klasörüne Gidin

Terminal/PowerShell'de proje klasörüne gidin:

```bash
cd C:\Users\pergem\Desktop\finalproje\finalProje
```

**Windows PowerShell için:**
```powershell
cd "C:\Users\pergem\Desktop\finalproje\finalProje"
```

### Adım 3: Sanal Ortam Oluşturun (Önerilir)

Sanal ortam oluşturmak proje bağımlılıklarını izole etmek için önerilir:

**Windows:**
```bash
python -m venv venv
```

**Sanal ortamı aktifleştirin:**
```bash
venv\Scripts\activate
```

Aktifleştirildiğinde terminalinizde `(venv)` yazısını görmelisiniz.

### Adım 4: Python Bağımlılıklarını Yükleyin

Proje klasöründeyken, `requirements.txt` dosyasındaki tüm paketleri yükleyin:

```bash
pip install -r requirements.txt
```

Bu komut şu paketleri yükleyecektir:
- fastapi==0.104.1
- uvicorn==0.24.0
- pandas==2.1.3
- numpy==1.25.2
- pydantic==2.5.0
- python-multipart==0.0.6
- jinja2==3.1.2

**Ek olarak, scikit-learn ve openpyxl paketlerini de yükleyin:**

```bash
pip install scikit-learn openpyxl
```

**Not:** Eğer yükleme sırasında hata alırsanız, pip'i güncelleyin:
```bash
python -m pip install --upgrade pip
```

---

## ▶️ Çalıştırma Adımları

### Adım 1: AI Analizini Çalıştırın

Öncelikle, makine öğrenmesi analizini çalıştırarak analiz sonuçlarını oluşturmanız gerekir. Bu adım, CSV sonuç dosyalarını oluşturacaktır.

**Terminal/PowerShell'de:**

```bash
python personnel_analysis_ai.py
```

**Beklenen Çıktı:**
- Analiz süreci başlayacak ve veri yükleme, ön işleme, model eğitimi ve tahmin adımları gerçekleşecek
- İşlem tamamlandığında şu dosyalar oluşturulacak:
  - `personnel_analysis_results.csv`
  - `veteriner_analysis_results.csv`
  - `gida_analysis_results.csv`
  - `ziraat_analysis_results.csv`

**Not:** Bu işlem birkaç dakika sürebilir. Analiz tamamlandığında terminalde "ANALYSIS COMPLETED SUCCESSFULLY!" mesajını göreceksiniz.

### Adım 2: FastAPI Sunucusunu Başlatın

Analiz tamamlandıktan sonra, web sunucusunu başlatın:

```bash
python app.py
```

veya

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Beklenen Çıktı:**
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Adım 3: Web Arayüzünü Açın

Sunucu başladıktan sonra, web tarayıcınızda şu adresi açın:

```
http://localhost:8000/dashboard
```

veya

```
http://127.0.0.1:8000/dashboard
```

**API Dokümantasyonu:**
FastAPI otomatik olarak API dokümantasyonu sağlar. Şu adresi ziyaret edebilirsiniz:

```
http://localhost:8000/docs
```

---

## 📊 Kullanım

### Dashboard Özellikleri

1. **Genel Bakış Kartları:**
   - Toplam İlçe Sayısı
   - Norm Fazlası İlçeler
   - Norm Eksiği İlçeler
   - Dengede Olan İlçeler

2. **Grafikler:**
   - Şehirlere Göre En Yüksek Norm Fazlası (Bar Chart)
   - Genel Durum Dağılımı (Pie Chart)

3. **Detaylı Liste:**
   - Tüm ilçeler için meslek bazında detaylı tablo
   - Arama ve filtreleme özellikleri
   - Sıralama özellikleri

### API Endpoint'leri

- `GET /api/summary` - Genel özet istatistikleri
- `GET /api/districts` - İlçe verilerini listeleme (filtreleme ve sayfalama destekli)
- `GET /api/districts/{il_adi}/{ilce_adi}` - Belirli bir ilçenin detaylı bilgileri
- `GET /api/search?query=...` - İlçe arama
- `GET /api/top-deficits/{profession}` - En yüksek eksikliğe sahip ilçeler
- `GET /api/top-surpluses/{profession}` - En yüksek fazlalığa sahip ilçeler

---

## 🔍 Sorun Giderme

### Problem 1: "ModuleNotFoundError" Hatası

**Çözüm:** Eksik paketleri yükleyin:
```bash
pip install -r requirements.txt
pip install scikit-learn openpyxl
```

### Problem 2: "Port 8000 already in use" Hatası

**Çözüm:** Farklı bir port kullanın:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

Sonra tarayıcıda `http://localhost:8001/dashboard` adresini açın.

### Problem 3: CSV Dosyaları Bulunamıyor

**Çözüm:** Önce `personnel_analysis_ai.py` scriptini çalıştırdığınızdan emin olun. Bu script analiz sonuçlarını oluşturur.

### Problem 4: Türkçe Karakter Sorunları

**Çözüm:** Terminal/PowerShell'in UTF-8 kodlamasını desteklediğinden emin olun. Windows PowerShell için:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### Problem 5: Excel Dosyası Okunamıyor

**Çözüm:** `openpyxl` paketinin yüklü olduğundan emin olun:
```bash
pip install openpyxl
```

### Problem 6: Veri Dosyaları Eksik

**Çözüm:** `data/` klasöründe tüm gerekli CSV ve Excel dosyalarının bulunduğundan emin olun:
- personel_durum.csv
- il_ilce_18yas_nufus.xlsx
- ilce_yuzolcum.csv
- norm_veteriner_hekim.csv
- norm_gida_mühendisi.csv
- norm_ziraat_mühendisi.csv
- tarim_alani.csv
- tarimsal_uretim.csv
- canli_hayvan.csv
- denetim_sayisi.csv

---

## 📝 Önemli Notlar

1. **İlk Çalıştırma:** İlk çalıştırmada mutlaka `personnel_analysis_ai.py` scriptini çalıştırın. Bu script analiz sonuçlarını oluşturur.

2. **Veri Güncellemesi:** Eğer veri dosyalarını güncellerseniz, analizi tekrar çalıştırmanız gerekir.

3. **Sunucu Durdurma:** Sunucuyu durdurmak için terminalde `CTRL+C` tuşlarına basın.

4. **Performans:** Analiz işlemi veri boyutuna bağlı olarak birkaç dakika sürebilir. Büyük veri setleri için daha uzun sürebilir.

5. **Tarayıcı Uyumluluğu:** Modern tarayıcılar (Chrome, Firefox, Edge, Safari) desteklenir.

---

## 🎯 Hızlı Başlangıç Özeti

```bash
# 1. Proje klasörüne git
cd C:\Users\pergem\Desktop\finalproje\finalProje

# 2. Sanal ortam oluştur (opsiyonel)
python -m venv venv
venv\Scripts\activate

# 3. Bağımlılıkları yükle
pip install -r requirements.txt
pip install scikit-learn openpyxl

# 4. AI analizini çalıştır
python personnel_analysis_ai.py

# 5. Sunucuyu başlat
python app.py

# 6. Tarayıcıda aç
# http://localhost:8000/dashboard
```

---

## 📞 Destek

Sorun yaşarsanız:
1. Terminal çıktılarını kontrol edin
2. Hata mesajlarını okuyun
3. Bu dokümandaki "Sorun Giderme" bölümüne bakın
4. Python ve paket sürümlerinizi kontrol edin

---

**İyi çalışmalar! 🚀**

