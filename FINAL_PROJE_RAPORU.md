# FİNAL PROJE RAPORU: TARIM PERSONELİ NORM KADRO TAHMİN SİSTEMİ

**Ders:** AIE521 - Derin Öğrenme
**Öğrenci:** Kerim GÜLLÜ
**Tarih:** Aralık 2024

---

## 1. YÖNETİCİ ÖZETİ

Bu proje, Türkiye genelindeki **916 ilçe** için tarım ve hayvancılık hizmetlerinin sürdürülebilirliği adına kritik öneme sahip olan **Veteriner Hekim**, **Gıda Mühendisi** ve **Ziraat Mühendisi** norm kadro sayılarını tahmin eden yapay zeka tabanlı bir sistemdir.

Klasik istatistiksel yöntemlerin aksine, bu çalışmada **PyTorch** kütüphanesi ile **MLP**, **ResNet** ve **AttentionNet** olmak üzere üç farklı derin öğrenme mimarisi sıfırdan kodlanmış, eğitilmiş ve performansları karşılaştırılmıştır. Veri setindeki dengesizlikleri gidermek için **Mixup** ve **SMOTE-like** veri artırma algoritmaları yine proje kapsamında kodlanarak uygulanmıştır.

### Temel Başarılar:
- **Gıda Mühendisi** modeli için **R² = 0.9156** (çok yüksek doğruluk)
- **Veteriner Hekim** modeli için **R² = 0.7327** (iyi doğruluk)
- **Ziraat Mühendisi** modeli için **R² = 0.6791** (kabul edilebilir doğruluk)

---

## 2. VERİ SETİ VE ÖZNİTELİK MÜHENDİSLİĞİ

Projede kullanılan veri seti, Tarım ve Orman Bakanlığı ile TÜİK kaynaklarından derlenen **916 ilçeye** ait kayıtlardan oluşmaktadır. Modelin başarısı için her meslek grubuna özgü, alana dayalı öznitelikler (features) seçilmiştir.

### 2.1. Veri Dosyaları

| Dosya Adı | Konum | Açıklama |
|-----------|-------|----------|
| `veteriner_hekim_birlesitirilmis_veri.csv` | `data/processed/` | Veteriner hekim verileri (916 kayıt) |
| `gida_muhendisi_birlesitirilmis_veri.csv` | `data/processed/` | Gıda mühendisi verileri (916 kayıt) |
| `ziraat_muhendisi_birlesitirilmis_veri.csv` | `data/processed/` | Ziraat mühendisi verileri (916 kayıt) |

### 2.2. Öznitelik Listesi (Feature Sets)

Hocamızın belirttiği gibi modelin girdilerini açıkça belirtmek gerekirse; her meslek grubu için kullanılan öznitelikler `src/dataset.py` dosyasında şu şekilde tanımlanmıştır:

#### A. Veteriner Hekim Modeli (28 Özellik)

**Kod Referansı:** `src/dataset.py` : Satır 265-283

Bu modelde hayvan varlığı ve sağlık taramaları odaklı şu öznitelikler kullanılmıştır:

| # | Öznitelik Adı | Açıklama | Kategori |
|---|---------------|----------|----------|
| 1 | `yuzolcum` | İlçe yüzölçümü (km²) | Coğrafi |
| 2 | `merkez_ilce_flag` | İl merkezi olup olmadığı (0/1) | Coğrafi |
| 3 | `nufus_18plus` | 18 yaş üstü nüfus | Demografik |
| 4 | `nufus_yogunlugu_km2` | Kilometrekareye düşen insan sayısı | Demografik |
| 5 | `toplam_hayvan_2021` | 2021 yılı toplam hayvan varlığı | Hayvan |
| 6 | `toplam_hayvan_2022` | 2022 yılı toplam hayvan varlığı | Hayvan |
| 7 | `toplam_hayvan_2023` | 2023 yılı toplam hayvan varlığı | Hayvan |
| 8 | `toplam_hayvan_2024` | 2024 yılı toplam hayvan varlığı | Hayvan |
| 9 | `toplam_hayvan_ortalama` | 4 yıllık hayvan ortalaması | Hayvan |
| 10 | `hayvan_nufus_orani` | Kişi başına düşen hayvan sayısı | Hayvan |
| 11 | `hayvan_yuzolcum_orani` | Birim alana düşen hayvan yoğunluğu | Hayvan |
| 12 | `tarim_alani_2020` | 2020 tarım alanı (dekar) | Tarım |
| 13 | `tarim_alani_2021` | 2021 tarım alanı (dekar) | Tarım |
| 14 | `tarim_alani_2022` | 2022 tarım alanı (dekar) | Tarım |
| 15 | `tarim_alani_2023` | 2023 tarım alanı (dekar) | Tarım |
| 16 | `tarim_alani_2024` | 2024 tarım alanı (dekar) | Tarım |
| 17 | `tarim_alani_ortalama` | Ortalama tarım alanı | Tarım |
| 18 | `tarim_alani_trend_5y` | Tarım alanı 5 yıllık değişim trendi | Tarım |
| 19 | `tarim_alani_yuzolcum_orani` | Tarım arazisinin ilçeye oranı | Tarım |
| 20 | `denetim_sayisi` | Toplam denetim sayısı | Denetim |
| 21 | `denetim_nufus_orani` | Nüfusa göre denetim yoğunluğu | Denetim |
| 22 | `denetim_hayvan_orani` | Hayvan başına düşen denetim | Denetim |
| 23 | `veteriner_islem_adet` | Sistemde kayıtlı veterinerlik işlemi sayısı | İşlem |
| 24 | `veteriner_islem_sure_toplam` | İşlemlerin aldığı toplam süre (saat) | İşlem |
| 25 | `veteriner_islem_nufus_orani` | Nüfusa oranla işlem yoğunluğu | İşlem |
| 26 | `veteriner_islem_hayvan_orani` | Hayvan başına işlem oranı | İşlem |
| 27 | `veteriner_islem_yuzolcum_orani` | Alana göre işlem yoğunluğu | İşlem |
| 28 | `veteriner_hekim_yas_ortalam` | Mevcut personelin yaş ortalaması | Personel |

**Kod:**
```python
# src/dataset.py : Satır 265-283
VETERINER_FEATURES = [
    # Temel demografik
    'yuzolcum', 'merkez_ilce_flag',
    'nufus_18plus', 'nufus_yogunlugu_km2',
    # Hayvan verileri (tum yillar)
    'toplam_hayvan_2021', 'toplam_hayvan_2022', 'toplam_hayvan_2023', 'toplam_hayvan_2024',
    'toplam_hayvan_ortalama',
    'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
    # Tarim alani
    'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022', 'tarim_alani_2023', 'tarim_alani_2024',
    'tarim_alani_ortalama', 'tarim_alani_trend_5y', 'tarim_alani_yuzolcum_orani',
    # Denetim
    'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
    # Veteriner islem verileri
    'veteriner_islem_adet', 'veteriner_islem_sure_toplam',
    'veteriner_islem_nufus_orani', 'veteriner_islem_hayvan_orani', 'veteriner_islem_yuzolcum_orani',
    # Personel verimlilik
    'veteriner_hekim_yas_ortalam', 'veteriner_hekim_verimi'
]
```

#### B. Gıda Mühendisi Modeli (35 Özellik)

**Kod Referansı:** `src/dataset.py` : Satır 285-306

Gıda güvenliği ve denetim odaklı öznitelikler:

| # | Öznitelik Adı | Açıklama | Kategori |
|---|---------------|----------|----------|
| 1-4 | Temel Demografik | Yüzölçümü, Nüfus vb. | Coğrafi/Demografik |
| 5-11 | Hayvan Varlığı | Et ve süt ürünleri potansiyeli için | Hayvan |
| 12-19 | Tarım Alanı | Bitkisel gıda üretimi için | Tarım |
| 20 | `tarimsal_uretim_2021` | 2021 toplam üretim (ton) | Üretim |
| 21 | `tarimsal_uretim_2022` | 2022 toplam üretim (ton) | Üretim |
| 22 | `tarimsal_uretim_2023` | 2023 toplam üretim (ton) | Üretim |
| 23 | `tarimsal_uretim_2024` | 2024 toplam üretim (ton) | Üretim |
| 24 | `tarimsal_uretim_ortalama` | Ortalama üretim | Üretim |
| 25 | `denetim_sayisi` | Gıda işletmesi denetim sayısı | Denetim |
| 26 | `denetim_nufus_orani` | Nüfusa göre denetim | Denetim |
| 27 | `denetim_hayvan_orani` | Hayvana göre denetim | Denetim |
| 28 | `gida_islem_adet` | Kayıtlı gıda işlemi sayısı | İşlem |
| 29 | `gida_islem_sure_toplam` | İşlem süresi (saat) | İşlem |
| 30 | `gida_islem_nufus_orani` | Nüfusa oranla işlem | İşlem |
| 31 | `gida_islem_hayvan_orani` | Hayvana oranla işlem | İşlem |
| 32 | `gida_islem_yuzolcum_orani` | Alana oranla işlem | İşlem |
| 33 | `gida_islem_uretim_orani` | Üretime oranla işlem sayısı | İşlem |
| 34 | `gida_muhendisi_yas_ortalam` | Personel yaş ortalaması | Personel |
| 35 | `gida_muhendisi_verimi` | Personel verimlilik skoru | Personel |

**Kod:**
```python
# src/dataset.py : Satır 285-306
GIDA_FEATURES = [
    # Temel demografik
    'yuzolcum', 'merkez_ilce_flag',
    'nufus_18plus', 'nufus_yogunlugu_km2',
    # Hayvan verileri
    'toplam_hayvan_2021', 'toplam_hayvan_2022', 'toplam_hayvan_2023', 'toplam_hayvan_2024',
    'toplam_hayvan_ortalama',
    'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
    # Tarim alani
    'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022', 'tarim_alani_2023', 'tarim_alani_2024',
    'tarim_alani_ortalama', 'tarim_alani_trend_5y', 'tarim_alani_yuzolcum_orani',
    # Tarimsal uretim
    'tarimsal_uretim_2021', 'tarimsal_uretim_2022', 'tarimsal_uretim_2023', 'tarimsal_uretim_2024',
    'tarimsal_uretim_ortalama',
    # Denetim
    'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
    # Gida islem verileri
    'gida_islem_adet', 'gida_islem_sure_toplam',
    'gida_islem_nufus_orani', 'gida_islem_hayvan_orani', 'gida_islem_yuzolcum_orani', 'gida_islem_uretim_orani',
    # Personel verimlilik
    'gida_muhendisi_yas_ortalam', 'gida_muhendisi_verimi'
]
```

#### C. Ziraat Mühendisi Modeli (35 Özellik)

**Kod Referansı:** `src/dataset.py` : Satır 308-330

Bitkisel üretim odaklı öznitelikler:

| # | Öznitelik Adı | Açıklama | Kategori |
|---|---------------|----------|----------|
| 1-4 | Temel Demografik | Coğrafi ve nüfus verileri | Coğrafi |
| 5-11 | Hayvan Varlığı | Gübre yönetimi vb. için | Hayvan |
| 12-19 | Tarım Alanı | **En kritik girdi grubu** | Tarım |
| 20-24 | Tarımsal Üretim | 2021-2024 üretim miktarları | Üretim |
| 25-27 | Denetim Verileri | Tarım denetimleri | Denetim |
| 28 | `ziraat_islem_adet` | ÇKS, destekleme vb. işlem sayısı | İşlem |
| 29 | `ziraat_islem_sure_toplam` | Toplam işlem süresi | İşlem |
| 30 | `ziraat_islem_nufus_orani` | Nüfusa oranla işlem | İşlem |
| 31 | `ziraat_islem_hayvan_orani` | Hayvana oranla işlem | İşlem |
| 32 | `ziraat_islem_yuzolcum_orani` | Alana oranla işlem | İşlem |
| 33 | `ziraat_islem_tarim_alani_orani` | Tarım alanı başına işlem | İşlem |
| 34 | `ziraat_islem_uretim_orani` | Üretime oranla işlem | İşlem |
| 35 | `ziraat_muhendisi_yas_ortalam` | Personel yaş ortalaması | Personel |

---

## 3. VERİ ÖN İŞLEME VE TEMİZLEME (DATA PREPROCESSING)

Veri setindeki kirliliği gidermek ve modeli eğitime hazırlamak için `src/dataset.py` dosyasında `prepare_features` metodunda şu işlemler kodlanmıştır.

### 3.1. Sonsuz Değerlerin Temizlenmesi

**Kod Referansı:** `src/dataset.py` : Satır 413-414

```python
# src/dataset.py : Satır 413-414
# Handle infinite values first
X = X.replace([np.inf, -np.inf], np.nan)
```

**Açıklama:** Bazı oran hesaplamalarında ortaya çıkan sonsuz değerler (`inf`, `-inf`) önce `NaN` değerine dönüştürülür.

### 3.2. Eksik Veri Doldurma (Median Imputation)

**Kod Referansı:** `src/dataset.py` : Satır 416-428

Veri setindeki `NaN` (boş) değerler, veri dağılımını bozmamak için sütunun **medyanı** ile doldurulmuştur:

```python
# src/dataset.py : Satır 416-428
# Calculate median for each column (ignoring NaN)
medians = X.median()

# Replace NaN with 0 if median is also NaN
for col in X.columns:
    if pd.isna(medians[col]):
        medians[col] = 0

# Fill missing values
X = X.fillna(medians)

# Double check - replace any remaining NaN with 0
X = X.fillna(0)
```

**Neden Medyan?** Ortalama (mean) aykırı değerlerden etkilenirken, medyan daha sağlam (robust) bir merkezi eğilim ölçüsüdür. Bu sayede veri dağılımı korunur.

### 3.3. Aykırı Değer Baskılama (Outlier Clipping)

**Kod Referansı:** `src/dataset.py` : Satır 430-436

Bazı ilçelerde (örn: büyükşehir merkezleri) işlem sayıları veya nüfus çok yüksek olduğu için modelin gradyanlarını bozabilmektedir. Bu nedenle, özellikle **oran** (`_orani`) ve **işlem** (`islem`) içeren sütunlar **%1 ve %99 yüzdelik dilimlerine** (percentile) sabitlenmiştir:

```python
# src/dataset.py : Satır 430-436
# Clip extreme outliers to 99th percentile for each column
# This prevents extreme values from dominating the model
for col in X.columns:
    if '_orani' in col or 'islem' in col:  # Ratio columns tend to have outliers
        q99 = X[col].quantile(0.99)
        q01 = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=q01, upper=q99)
```

**Matematiksel Açıklama:**
- `q01` = Verinin %1'lik alt sınırı
- `q99` = Verinin %99'luk üst sınırı
- `clip(lower, upper)` = Değerleri bu aralıkta tutar

### 3.4. Hedef Değişken Temizleme

**Kod Referansı:** `src/dataset.py` : Satır 438-441

```python
# src/dataset.py : Satır 438-441
# IMPORTANT: Fill NaN target values with 0
# In personel_durum.csv, empty values mean "no personnel" (0), not missing data
# This is verified by checking that yas_ortalam and verimi are also 0 for these rows
y = y.fillna(0)
```

**Açıklama:** Personel sayısındaki boş değerler, "personel yok" anlamına geldiği için 0 ile doldurulmuştur.

### 3.5. Normalizasyon (StandardScaler)

**Kod Referansı:** `src/dataset.py` : Satır 507-510

Tüm özellikler `sklearn.preprocessing.StandardScaler` kullanılarak normalize edilmiştir:

```python
# src/dataset.py : Satır 507-510
# Scale features (fit on training data)
X_train_scaled = self.scaler.fit_transform(X_train)
X_val_scaled = self.scaler.transform(X_val)
X_test_scaled = self.scaler.transform(X_test)
```

**Matematiksel Formül:**
```
z = (x - μ) / σ
```
- `μ` = Sütun ortalaması
- `σ` = Sütun standart sapması
- `z` = Normalize edilmiş değer (ortalama=0, std=1)

**Önemli Not:** Scaler sadece eğitim verisi üzerinde `fit` edilir, validation ve test verisine sadece `transform` uygulanır. Bu, veri sızıntısını (data leakage) önler.

### 3.6. Veri Bölme (Train/Validation/Test Split)

**Kod Referansı:** `src/dataset.py` : Satır 486-494

```python
# src/dataset.py : Satır 486-494
# First split: train+val vs test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Second split: train vs val (90/10 of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_size, random_state=random_state
)
```

**Veri Dağılımı:**

| Set | Oran | Örnek Sayısı | Kullanım Amacı |
|-----|------|--------------|----------------|
| Eğitim (Train) | %70 | 658 | Model ağırlıklarını öğrenmek |
| Doğrulama (Validation) | %10 | 74 | Hiperparametre ayarlama, early stopping |
| Test | %20 | 184 | Final performans değerlendirme |

---

## 4. VERİ ARTIRMA (DATA AUGMENTATION)

Veri seti küçük olduğu için (916 örnek), eğitim sırasında sentetik veri üretimi yapılmıştır. `src/dataset.py` içindeki `DataAugmenter` sınıfı şu teknikleri uygular:

### 4.1. Gaussian Noise (Gürültü Ekleme)

**Kod Referansı:** `src/dataset.py` : Satır 46-79

```python
# src/dataset.py : Satır 67-77
# Calculate feature-wise standard deviation
feature_std = np.std(X, axis=0) + 1e-8

# Add noise proportional to feature std
noise = np.random.normal(0, self.noise_level, X_selected.shape) * feature_std
X_augmented = X_selected + noise

# Add small noise to targets as well (proportional to target std)
target_std = np.std(y) + 1e-8
y_noise = np.random.normal(0, self.noise_level * 0.5, y_selected.shape) * target_std
y_augmented = np.maximum(y_selected + y_noise, 0.1)  # Keep positive
```

**Açıklama:** Her özelliğe, o özelliğin standart sapmasıyla orantılı Gaussian gürültü eklenir.

### 4.2. SMOTE-like (Sentetik Örnek Üretimi)

**Kod Referansı:** `src/dataset.py` : Satır 81-123

```python
# src/dataset.py : Satır 97-121
# Fit nearest neighbors
k = min(self.n_neighbors, len(X) - 1)
nn = NearestNeighbors(n_neighbors=k + 1)
nn.fit(X)

for _ in range(n_samples):
    # Randomly select a sample
    idx = np.random.randint(len(X))
    sample = X[idx]
    target = y[idx]

    # Find neighbors
    distances, neighbors = nn.kneighbors([sample])
    neighbor_idx = neighbors[0, np.random.randint(1, k + 1)]

    # Interpolate between sample and neighbor
    alpha = np.random.uniform(0.1, 0.9)
    new_sample = sample + alpha * (X[neighbor_idx] - sample)
    new_target = target + alpha * (y[neighbor_idx] - target)
```

**Matematiksel Formül:**
```
X_new = X_i + α × (X_neighbor - X_i)
y_new = y_i + α × (y_neighbor - y_i)
```
- `α` ∈ [0.1, 0.9] rastgele bir interpolasyon katsayısı

### 4.3. Mixup Augmentation

**Kod Referansı:** `src/dataset.py` : Satır 125-159

```python
# src/dataset.py : Satır 145-157
for _ in range(n_samples):
    # Select two random samples
    idx1, idx2 = np.random.choice(len(X), 2, replace=False)

    # Sample mixing coefficient from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Mix samples
    new_X = lam * X[idx1] + (1 - lam) * X[idx2]
    new_y = lam * y[idx1] + (1 - lam) * y[idx2]

    mixed_X.append(new_X)
    mixed_y.append(max(new_y, 0.1))
```

**Matematiksel Formül:**
```
X_new = λ × X_i + (1-λ) × X_j
y_new = λ × y_i + (1-λ) × y_j
```
- `λ` ~ Beta(α, α) dağılımından örneklenir (varsayılan α=0.4)

### 4.4. Feature Jittering

**Kod Referansı:** `src/dataset.py` : Satır 161-178

```python
# src/dataset.py : Satır 173-176
# Random multiplicative jitter
jitter = 1 + np.random.uniform(-jitter_ratio, jitter_ratio, X.shape)
X_jittered = X * jitter
```

**Açıklama:** Her özellik değeri, ±%5 oranında rastgele çarpılır.

---

## 5. GELİŞTİRİLEN DERİN ÖĞRENME MİMARİLERİ

Projede `src/model.py` dosyasında tanımlanan üç farklı mimari kullanılmıştır.

### 5.1. MLP (Multi-Layer Perceptron)

**Kod Referansı:** `src/model.py` : Satır 18-73

Tablosal verilerde genellikle en iyi sonucu veren, **Batch Normalization** ve **Dropout** ile güçlendirilmiş tam bağlantılı ağdır.

**Mimari Diyagramı:**
```
Input(28-35) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
             → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
             → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
             → Linear(1)   → Output
```

**Kod Yapısı:**
```python
# src/model.py : Satır 44-60
# Build layers
layers = []
prev_dim = input_dim

for hidden_dim in hidden_dims:
    layers.extend([
        nn.Linear(prev_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate)
    ])
    prev_dim = hidden_dim

# Output layer (single value for regression)
layers.append(nn.Linear(prev_dim, 1))

self.network = nn.Sequential(*layers)
```

**Forward Pass:**
```python
# src/model.py : Satır 62-72
def forward(self, x):
    """
    Forward pass through the network.

    Args:
        x: Input tensor of shape (batch_size, input_dim)

    Returns:
        Output tensor of shape (batch_size, 1)
    """
    return self.network(x).squeeze(-1)
```

**Parametre Sayısı:** ~14,500 - 15,500

### 5.2. ResNet (Residual Network)

**Kod Referansı:** `src/model.py` : Satır 75-163

Derin ağlarda **gradyan kaybolmasını** önlemek için **"skip connection"** (atlamalı bağlantı) kullanan mimaridir. Giriş verisi, işlem bloğunun çıkışına eklenir.

**Mimari Diyagramı:**
```
Input → Linear(128) → BatchNorm → ReLU
      → [ResidualBlock × 3]
      → Linear(1) → Output

ResidualBlock:
    x → Linear → BN → ReLU → Dropout → Linear → BN → (+x) → ReLU
                                                  ↑
                                            Skip Connection
```

**Residual Block Kodu:**
```python
# src/model.py : Satır 83-109
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.2):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass with skip connection."""
        residual = x            # Girişi sakla
        out = self.block(x)     # Katmanlardan geçir
        out = out + residual    # Skip connection: Girişi çıkışa ekle
        out = self.relu(out)
        return out
```

**Matematiksel Formül:**
```
y = F(x) + x
```
- `F(x)` = Blok içindeki dönüşüm
- `x` = Skip connection ile eklenen orijinal girdi

**Parametre Sayısı:** ~104,700 - 105,600

### 5.3. AttentionNet (Self-Attention)

**Kod Referansı:** `src/model.py` : Satır 165-263

Modelin hangi özelliğin (örn: hayvan sayısı mı, yüzölçümü mü?) daha önemli olduğunu öğrenmesini sağlayan **dikkat mekanizmasıdır**.

**Mimari Diyagramı:**
```
Input → Embedding(128) → Self-Attention → MLP → Output
```

**Attention Mekanizması Kodu:**
```python
# src/model.py : Satır 173-211
class AttentionLayer(nn.Module):
    def __init__(self, dim: int):
        super(AttentionLayer, self).__init__()

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # For 1D input, we treat each feature as a "token"
        x = x.unsqueeze(1)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Attention weights
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        return out.squeeze(1)
```

**Matematiksel Formül (Scaled Dot-Product Attention):**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- `Q` = Query matrisi
- `K` = Key matrisi
- `V` = Value matrisi
- `d_k` = Key boyutu (ölçekleme için)

**Parametre Sayısı:** ~63,800 - 64,700

### 5.4. Model Factory Fonksiyonu

**Kod Referansı:** `src/model.py` : Satır 265-287

```python
# src/model.py : Satır 265-287
def get_model(model_name: str, input_dim: int, **kwargs):
    """
    Factory function to create models by name.

    Args:
        model_name: One of 'mlp', 'resnet', 'attention'
        input_dim: Number of input features
        **kwargs: Additional model parameters

    Returns:
        Initialized model
    """
    models = {
        'mlp': PersonnelMLP,
        'resnet': PersonnelResNet,
        'attention': PersonnelAttentionNet
    }

    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Available: {list(models.keys())}")

    return models[model_name.lower()](input_dim, **kwargs)
```

---

## 6. EĞİTİM SÜRECİ

### 6.1. Eğitim Konfigürasyonu

**Kod Referansı:** `train.py` : Satır 411-430

| Parametre | Değer | Kod Satırı | Açıklama |
|-----------|-------|------------|----------|
| Kayıp Fonksiyonu | MSE Loss | `train.py:413` | Regresyon için standart |
| Optimizer | Adam | `train.py:417-421` | Adaptif öğrenme oranlı |
| Öğrenme Oranı | 0.001 | `train.py:99` | Başlangıç LR |
| Weight Decay | 1e-4 | `train.py:101-102` | L2 regularizasyon |
| Batch Size | 32 | `train.py:97-98` | Mini-batch boyutu |
| Max Epoch | 100 | `train.py:95-96` | Maksimum epoch |
| Early Stopping | 15 | `train.py:103-104` | Patience değeri |

**Kayıp Fonksiyonu (MSE Loss):**
```python
# train.py : Satır 413
criterion = nn.MSELoss()
```

**Matematiksel Formül:**
```
MSE = (1/n) × Σ(y_pred - y_true)²
```

**Adam Optimizer:**
```python
# train.py : Satır 417-421
optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,                    # Öğrenme oranı
    weight_decay=args.weight_decay  # L2 regularizasyon
)
```

### 6.2. Öğrenme Oranı Zamanlayıcı (Learning Rate Scheduler)

**Kod Referansı:** `train.py` : Satır 423-430

```python
# train.py : Satır 423-430
# ReduceLROnPlateau: Öğrenme oranı zamanlayıcısı
# Validation loss iyileşmezse öğrenme oranını düşürür
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # Kayıp azaldığında iyileşme var
    factor=0.5,      # LR'yi yarıya düşür
    patience=5       # 5 epoch iyileşme olmazsa düşür
)
```

**Çalışma Mantığı:**
- Validation loss 5 epoch boyunca iyileşmezse → LR = LR × 0.5
- Bu sayede model platoya ulaştığında daha ince ayar yapabilir

### 6.3. Gradyan Kırpma (Gradient Clipping)

**Kod Referansı:** `train.py` : Satır 218

```python
# train.py : Satır 218
# Patlayan gradyan problemini önlemek için gradyanları sınırla
# max_norm=1.0: Gradyan vektörünün normu 1'i aşamaz
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Neden Gerekli?** Derin ağlarda gradyanlar çok büyüyebilir (exploding gradients), bu da eğitimi bozar. Gradient clipping bu sorunu önler.

### 6.4. Early Stopping (Erken Durdurma)

**Kod Referansı:** `train.py` : Satır 488-520

```python
# train.py : Satır 488-520
if val_loss < best_val_loss:
    # Yeni en iyi model bulundu!
    best_val_loss = val_loss
    best_val_r2 = val_r2
    patience_counter = 0  # Sayacı sıfırla

    # En iyi modeli kaydet
    model_path = os.path.join(
        args.save_dir,
        f'{args.profession}_{args.model}_best.pt'
    )

    # Checkpoint kaydet - model ağırlıkları + metadata
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'input_dim': input_dim,
        'model_type': args.model,
        'profession': args.profession,
        'scaler': preprocessor.get_scaler(),
        'feature_names': preprocessor.get_feature_names()
    }, model_path)
else:
    # İyileşme yok, early stopping sayacını artır
    patience_counter += 1
    if patience_counter >= args.patience:
        # Patience aşıldı, eğitimi erken durdur
        print(f"Erken durdurma (Early Stopping) - Epoch {epoch+1}")
        break
```

**Çalışma Mantığı:**
1. Validation loss iyileşirse → En iyi modeli kaydet, sayacı sıfırla
2. İyileşmezse → Sayacı artır
3. Sayaç ≥ 15 olursa → Eğitimi durdur

---

## 7. DEĞERLENDİRME METRİKLERİ

**Kod Referansı:** `test.py` : Satır 189-235

### 7.1. MAE (Mean Absolute Error - Ortalama Mutlak Hata)

```python
# test.py : Satır 196
mae = np.mean(np.abs(predictions - targets))
```

**Formül:** `MAE = (1/n) × Σ|y_pred - y_true|`

**Yorum:** Ortalama kaç personel hata yapıyoruz? (Örn: MAE=1.69 → ortalama 1.69 kişi sapma)

### 7.2. MSE (Mean Squared Error - Ortalama Kare Hata)

```python
# test.py : Satır 201
mse = np.mean((predictions - targets) ** 2)
```

**Formül:** `MSE = (1/n) × Σ(y_pred - y_true)²`

**Yorum:** Büyük hataları daha fazla cezalandırır.

### 7.3. RMSE (Root Mean Squared Error - Kök Ortalama Kare Hata)

```python
# test.py : Satır 206
rmse = np.sqrt(mse)
```

**Formül:** `RMSE = √MSE`

**Yorum:** MAE ile aynı birimde (personel sayısı), büyük hatalara daha duyarlı.

### 7.4. R² (R-kare - Determinasyon Katsayısı)

```python
# test.py : Satır 212-214
ss_res = np.sum((targets - predictions) ** 2)  # Kalan kareler toplamı
ss_tot = np.sum((targets - np.mean(targets)) ** 2)  # Toplam kareler toplamı
r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
```

**Formül:** `R² = 1 - (SS_res / SS_tot)`

**Yorum:**
- R² = 1.0 → Mükemmel tahmin
- R² = 0.0 → Ortalama kadar iyi (model anlamsız)
- R² < 0 → Ortalamadan kötü

### 7.5. MAPE (Mean Absolute Percentage Error - Ortalama Mutlak Yüzde Hata)

```python
# test.py : Satır 220-225
non_zero_mask = targets != 0  # Sıfır olmayan hedefler
if non_zero_mask.any():
    mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask])
                          / targets[non_zero_mask])) * 100
else:
    mape = 0
```

**Formül:** `MAPE = (1/n) × Σ|(y_true - y_pred) / y_true| × 100`

**Yorum:** Yüzde olarak hata oranı.

---

## 8. DENEYSEL SONUÇLARIN KARŞILAŞTIRILMASI

### 8.1. Tüm Modellerin Test Seti Performansı

**Script:** `python test.py --all`

| Meslek | Model | Test R² | Test MAE | RMSE | Parametre Sayısı |
|--------|-------|---------|----------|------|------------------|
| **Gıda Mühendisi** | **MLP** | **0.9156** | **0.38** | 0.67 | ~15,425 |
| | ResNet | 0.8965 | 0.41 | 0.75 | ~105,601 |
| | Attention | 0.8695 | 0.49 | 0.84 | ~64,705 |
| **Veteriner Hekim** | **MLP** | **0.7327** | **1.69** | 2.36 | ~14,529 |
| | ResNet | 0.7111 | 1.72 | 2.45 | ~104,705 |
| | Attention | 0.7269 | 1.70 | 2.38 | ~63,809 |
| **Ziraat Mühendisi** | **MLP** | **0.6791** | **2.49** | 3.68 | ~15,425 |
| | ResNet | 0.6668 | 2.54 | 3.75 | ~105,601 |
| | Attention | 0.6482 | 2.64 | 3.85 | ~64,705 |

**Analiz:** Beklentinin aksine, daha az parametreye sahip olan **MLP**, tablosal verilerde aşırı öğrenmeye (overfitting) daha az meyilli olduğu için karmaşık ResNet ve Attention modellerini geride bırakmıştır.

### 8.2. Data Augmentation Karşılaştırması

**Script:** `python compare_augmentation.py`

| Meslek | Yöntem | R² Skoru | Değişim |
|--------|--------|----------|---------|
| **Veteriner** | Baseline (Augmentation YOK) | 0.7099 | - |
| | Gaussian Noise | 0.7192 | +0.0093 |
| | SMOTE-like | 0.6994 | -0.0105 |
| | **Mixup** | **0.7231** | **+0.0132** |
| | Tüm Yöntemler | 0.6943 | -0.0156 |
| **Gıda** | Baseline (Augmentation YOK) | 0.9150 | - |
| | Gaussian Noise | 0.9005 | -0.0145 |
| | **SMOTE-like** | **0.9163** | **+0.0013** |
| | Mixup | 0.8799 | -0.0351 |
| | Tüm Yöntemler | 0.8919 | -0.0231 |
| **Ziraat** | Baseline (Augmentation YOK) | 0.6850 | - |
| | Gaussian Noise | 0.6804 | -0.0046 |
| | **SMOTE-like** | **0.6916** | **+0.0066** |
| | Mixup | 0.6728 | -0.0122 |
| | Tüm Yöntemler | 0.6714 | -0.0136 |

**Yorum:**
- **Veteriner** verisi gibi karmaşık dağılıma sahip veri setlerinde **Mixup** en iyi sonucu vermiştir.
- **Gıda** ve **Ziraat** için **SMOTE-like** yöntemi marjinal iyileşme sağlamıştır.

### 8.3. External Test (Genelleme) Sonuçları

**Script:** `python external_test.py`

Model, eğitim setinde hiç görmediği veriler üzerinde test edilmiştir.

#### Yöntem 1: İl Bazlı (10 il tamamen dışarıda)

**Dışarıda Tutulan İller:** Trabzon, Şanlıurfa, Eskişehir, Muğla, Kayseri, Samsun, Mardin, Çanakkale, Erzurum, Adana

**Kod Referansı:** `external_test.py` : Satır 55-62

| Meslek | Internal R² | External R² | Fark | Sonuç |
|--------|-------------|-------------|------|-------|
| Veteriner | 0.6608 | 0.7456 | +0.0848 | ✓ Daha iyi genelleme |
| Gıda | 0.8529 | 0.7641 | -0.0888 | ~ Kabul edilebilir düşüş |
| Ziraat | 0.7023 | 0.6572 | -0.0451 | ~ Stabil |

#### Yöntem 2: Hold-out (%20 ilçe tamamen dışarıda)

| Meslek | Internal R² | External R² | Fark | Sonuç |
|--------|-------------|-------------|------|-------|
| Veteriner | 0.6833 | 0.6644 | -0.0190 | ~ Stabil |
| Gıda | 0.7979 | 0.8980 | +0.1000 | ✓ Daha iyi genelleme |
| Ziraat | 0.6642 | 0.6641 | -0.0001 | ✓ Çok stabil |

**Genel Değerlendirme:** Model, görmediği verilerde de tutarlı performans göstermiştir. Internal ve External R² arasındaki farkların küçük olması, modelin iyi genelleme yaptığını göstermektedir.

---

## 9. PROJE DOSYA YAPISI

```
tarim/
├── train.py                        # Model eğitim scripti (677 satır)
├── test.py                         # Model test scripti (631 satır)
├── compare_augmentation.py         # Data augmentation karşılaştırma (298 satır)
├── external_test.py                # External test seti değerlendirme (554 satır)
├── app.py                          # FastAPI web sunucu
├── create_merged_datasets.py       # Veri birleştirme scripti
├── requirements.txt                # Python bağımlılıkları
├── README.md                       # Proje dokümantasyonu
├── TEKNIK_RAPOR.md                 # Teknik rapor (kısa versiyon)
├── FINAL_PROJE_RAPORU.md           # Bu dosya
│
├── src/                            # Kaynak kod modülleri
│   ├── __init__.py
│   ├── model.py                    # Sinir ağı mimarileri (334 satır)
│   └── dataset.py                  # Veri yükleme ve ön işleme (588 satır)
│
├── data/                           # Veri dosyaları
│   └── processed/                  # İşlenmiş birleşik veriler
│       ├── veteriner_hekim_birlesitirilmis_veri.csv
│       ├── gida_muhendisi_birlesitirilmis_veri.csv
│       └── ziraat_muhendisi_birlesitirilmis_veri.csv
│
├── checkpoints/                    # Eğitilmiş model dosyaları (.pt)
│   ├── veteriner_mlp_best.pt
│   ├── veteriner_resnet_best.pt
│   ├── veteriner_attention_best.pt
│   ├── gida_mlp_best.pt
│   ├── gida_resnet_best.pt
│   ├── gida_attention_best.pt
│   ├── ziraat_mlp_best.pt
│   ├── ziraat_resnet_best.pt
│   ├── ziraat_attention_best.pt
│   └── *_history.json              # Eğitim geçmişleri
│
├── results/                        # Test ve analiz sonuçları
│   ├── augmentation_comparison_results.csv
│   ├── external_test_results.csv
│   └── *_test_results.json
│
└── dashboard/                      # Web arayüzü dosyaları
    ├── index.html
    ├── app.js
    └── style.css
```

---

## 10. KULLANIM KILAVUZU

### 10.1. Tüm Modelleri Eğitme

```bash
python train.py --all --epochs 100
```

**Çıktı:** 9 model eğitilir (3 meslek × 3 mimari)

### 10.2. Tek Model Eğitme

```bash
python train.py --profession veteriner --model mlp --epochs 100
```

### 10.3. Tüm Modelleri Test Etme

```bash
python test.py --all
```

### 10.4. Data Augmentation Karşılaştırması

```bash
python compare_augmentation.py
```

### 10.5. External Test

```bash
python external_test.py
```

---

## 11. KULLANILAN KÜTÜPHANELERİ

| Kütüphane | Kullanım Amacı |
|-----------|----------------|
| **PyTorch** | Derin öğrenme framework |
| **pandas** | Veri manipülasyonu |
| **numpy** | Sayısal hesaplamalar |
| **scikit-learn** | StandardScaler, train_test_split, NearestNeighbors |
| **matplotlib** | Görselleştirme |
| **FastAPI** | Web API (opsiyonel) |

---

## 12. SONUÇ VE DEĞERLENDİRME

Bu projede, tarım personeli planlaması için uçtan uca bir derin öğrenme sistemi geliştirilmiştir.

### Temel Bulgular:

1. **MLP Mimarisi** tablosal verilerde ResNet ve AttentionNet'e göre daha yüksek performans göstermiştir. Bunun nedeni:
   - Daha az parametre → Daha az overfitting riski
   - Tablosal veriler için karmaşık mimariler gereksiz

2. **Veri Temizleme** aşamasında uygulanan:
   - Median imputation (eksik değer doldurma)
   - Outlier clipping (%1-%99 persentil)
   - StandardScaler normalizasyonu

   modelin kararlılığını artırmıştır.

3. **Data Augmentation** etkileri:
   - Veteriner için **Mixup** en iyi sonucu verdi (+1.9%)
   - Gıda ve Ziraat için **SMOTE-like** marjinal iyileşme sağladı

4. **External Test** sonuçları modelin iyi genelleme yaptığını doğrulamıştır.

### Model Performans Özeti:

| Meslek | En İyi Model | R² | MAE |
|--------|--------------|-----|-----|
| Gıda Mühendisi | MLP | **0.9156** | 0.38 |
| Veteriner Hekim | MLP | **0.7327** | 1.69 |
| Ziraat Mühendisi | MLP | **0.6791** | 2.49 |

---

*Bu rapor, proje kodlarından çıkarılmış detaylı referanslar içermektedir.*
*Tüm kod dosyaları ve eğitilmiş modeller GitHub reposunda mevcuttur.*
