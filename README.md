# Tarım Personeli Norm Tahmin Sistemi

Derin öğrenme kullanarak Türkiye'deki ilçeler için optimal tarım personeli sayısını tahmin eden sistem.

**Ders:** AIE521 - Derin Öğrenme
**Tarih:** Aralık 2024

---

## Proje Özeti

Bu proje, Türkiye'deki **916 ilçe** için üç farklı tarım personeli türünün optimal sayısını derin öğrenme modelleri ile tahmin etmektedir:

- **Veteriner Hekim** - Hayvan sağlığı hizmetleri
- **Gıda Mühendisi** - Gıda güvenliği denetimleri
- **Ziraat Mühendisi** - Tarımsal danışmanlık hizmetleri

---

## Dizin Yapısı

```
tarim/
├── data/                           # Veri dosyaları
│   ├── canli_hayvan.csv            # Canlı hayvan verileri (2020-2024)
│   ├── tarim_alani.csv             # Tarım alanı verileri
│   ├── tarimsal_uretim.csv         # Tarımsal üretim verileri
│   ├── denetim_sayisi.csv          # Denetim sayıları
│   ├── personel_durum.csv          # Mevcut personel durumu
│   ├── ilce_yuzolcum.csv           # İlçe coğrafi verileri
│   ├── norm_*.csv                  # Meslek bazlı işlem verileri
│   └── processed/                  # İşlenmiş birleşik veriler
│       ├── veteriner_hekim_birlesitirilmis_veri.csv
│       ├── gida_muhendisi_birlesitirilmis_veri.csv
│       └── ziraat_muhendisi_birlesitirilmis_veri.csv
│
├── src/                            # Kaynak kod modülleri
│   ├── __init__.py
│   ├── model.py                    # Sinir ağı mimarileri (MLP, ResNet, Attention)
│   └── dataset.py                  # Veri yükleme, ön işleme, augmentation
│
├── checkpoints/                    # Eğitilmiş model dosyaları (.pt)
│   ├── veteriner_mlp_best.pt
│   ├── gida_mlp_best.pt
│   ├── ziraat_mlp_best.pt
│   └── *_history.json              # Eğitim geçmişleri
│
├── results/                        # Test ve analiz sonuçları
│   ├── personnel_analysis_results.csv
│   ├── augmentation_comparison_results.csv
│   ├── external_test_results.csv
│   └── *_evaluation.png            # Görselleştirme grafikleri
│
├── dashboard/                      # Web arayüzü dosyaları
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── data.js
│
├── train.py                        # Model eğitim scripti
├── test.py                         # Model test scripti
├── compare_augmentation.py         # Data augmentation karşılaştırma
├── external_test.py                # External test seti değerlendirme
├── app.py                          # FastAPI web sunucu
├── create_merged_datasets.py       # Veri birleştirme scripti
├── requirements.txt                # Python bağımlılıkları
└── README.md                       # Bu dosya
```

---

## Kurulum

### 1. Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### 2. Gereksinimler
- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- FastAPI, uvicorn (dashboard için)

---

## Kullanım

### Model Eğitimi
```bash
# Tüm meslekler ve tüm modeller (3x3=9 model)
python train.py --all

# Tek meslek ve model eğit
python train.py --profession veteriner --model mlp --epochs 100

# Data augmentation ile eğit
python train.py --profession gida --augment --epochs 100

# Farklı model mimarisi
python train.py --profession ziraat --model resnet --epochs 150
```

### Model Testi
```bash
# Tüm meslekleri ve modelleri test et
python test.py --all

# Tek meslek ve model testi
python test.py --profession veteriner --model mlp

# Görselleştirme ile
python test.py --profession gida --model mlp --visualize
```

### Data Augmentation Karşılaştırması
```bash
python compare_augmentation.py
```

### External Test
```bash
python external_test.py
```

### Web Dashboard
```bash
python app.py
# Tarayıcıda: http://localhost:8080/dashboard
```

---

## Model Mimarileri

### 1. MLP (Multi-Layer Perceptron)
```
Input(35) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
         → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
         → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
         → Linear(1)   → Output
```

### 2. ResNet (Residual Network)
Artık bağlantılar (skip connections) ile gradient akışını iyileştirir.

### 3. AttentionNet
Self-attention mekanizması ile özellik ağırlıklandırma yapar.

---

## Sonuçlar

### Internal Test Sonuçları

| Meslek | Model | R² | MAE | RMSE |
|--------|-------|-----|-----|------|
| **Gıda Mühendisi** | MLP | **0.9256** | 0.36 | 0.63 |
| **Veteriner Hekim** | MLP | **0.7175** | 1.71 | 2.42 |
| **Ziraat Mühendisi** | MLP | **0.6839** | 2.46 | 3.65 |

### Data Augmentation Karşılaştırması

| Meslek | Baseline R² | En İyi Yöntem | En İyi R² | İyileşme |
|--------|-------------|---------------|-----------|----------|
| Veteriner | 0.7099 | Mixup | 0.7231 | +1.9% |
| Gıda | 0.9150 | SMOTE-like | 0.9163 | +0.1% |
| Ziraat | 0.6850 | SMOTE-like | 0.6916 | +1.0% |

### External Test Sonuçları

#### Yöntem 1: İl Bazlı (10 il tamamen ayrıldı)
| Meslek | Internal R² | External R² | Fark |
|--------|-------------|-------------|------|
| Veteriner | 0.6608 | 0.7456 | +0.0848 |
| Gıda | 0.8529 | 0.7641 | -0.0888 |
| Ziraat | 0.7023 | 0.6572 | -0.0451 |

#### Yöntem 2: Hold-out (%20 ilçe tamamen ayrıldı)
| Meslek | Internal R² | External R² | Fark |
|--------|-------------|-------------|------|
| Veteriner | 0.6833 | 0.6644 | -0.0190 |
| Gıda | 0.7979 | 0.8980 | +0.1000 |
| Ziraat | 0.6642 | 0.6641 | -0.0001 |

---

## Data Augmentation Yöntemleri

| Yöntem | Açıklama |
|--------|----------|
| **Gaussian Noise** | Özelliklere rastgele gürültü ekleme |
| **SMOTE-like** | En yakın komşu interpolasyonu ile sentetik örnek |
| **Mixup** | İki örneğin lineer kombinasyonu |
| **Feature Jittering** | Küçük çarpımsal pertürbasyonlar |

---

## Özellikler (Features)

### Kullanılan Özellikler (28-35 adet)

| Kategori | Özellikler |
|----------|-----------|
| **Coğrafi** | yuzolcum, merkez_ilce_flag |
| **Demografik** | nufus_18plus, nufus_yogunlugu_km2 |
| **Hayvan** | toplam_hayvan_2021-2024, hayvan_nufus_orani |
| **Tarım** | tarim_alani_2020-2024, tarim_alani_trend_5y |
| **Üretim** | tarimsal_uretim_2021-2024 |
| **Denetim** | denetim_sayisi, denetim_nufus_orani |
| **İşlem** | *_islem_adet, *_islem_sure_toplam |

---

## API Endpoints

| Endpoint | Açıklama |
|----------|----------|
| `GET /` | API bilgileri |
| `GET /api/summary` | Genel özet istatistikler |
| `GET /api/districts` | İlçe listesi (filtrelenebilir) |
| `GET /api/search?query=xxx` | İlçe arama |
| `GET /api/top-deficits/{meslek}` | En çok eksik olan ilçeler |
| `GET /api/top-surpluses/{meslek}` | En çok fazla olan ilçeler |
| `GET /dashboard` | Web arayüzü |

---

## Lisans

Bu proje AIE521 - Derin Öğrenme dersi kapsamında geliştirilmiştir.
