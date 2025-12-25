# Tarım Personeli Norm Tahmin Sistemi - Teknik Rapor

**Proje:** AIE521 - Derin Öğrenme Final Projesi
**Yazar:** Kerim GÜLLÜ
**Tarih:** Aralık 2024

---

## 1. Proje Özeti

Bu proje, Türkiye'deki 916 ilçe için üç farklı tarım personeli türünün (Veteriner Hekim, Gıda Mühendisi, Ziraat Mühendisi) optimal sayısını derin öğrenme modelleri ile tahmin etmektedir.

---

## 2. Veri Seti

### 2.1 Veri Kaynakları
- **916 ilçe** için demografik, tarımsal ve personel verileri
- Veri dosyaları: `data/processed/` klasöründe

| Dosya | Açıklama |
|-------|----------|
| `veteriner_hekim_birlesitirilmis_veri.csv` | Veteriner hekim verileri |
| `gida_muhendisi_birlesitirilmis_veri.csv` | Gıda mühendisi verileri |
| `ziraat_muhendisi_birlesitirilmis_veri.csv` | Ziraat mühendisi verileri |

### 2.2 Özellikler (Features)

| Meslek | Özellik Sayısı | Kod Referansı |
|--------|----------------|---------------|
| Veteriner Hekim | 28 | `src/dataset.py:265-283` |
| Gıda Mühendisi | 35 | `src/dataset.py:285-306` |
| Ziraat Mühendisi | 35 | `src/dataset.py:308-330` |

**Özellik Kategorileri:**
- Coğrafi: `yuzolcum`, `merkez_ilce_flag`
- Demografik: `nufus_18plus`, `nufus_yogunlugu_km2`
- Hayvan verileri: `toplam_hayvan_2021-2024`, `hayvan_nufus_orani`
- Tarım alanı: `tarim_alani_2020-2024`, `tarim_alani_trend_5y`
- Denetim: `denetim_sayisi`, `denetim_nufus_orani`
- İşlem verileri: `*_islem_adet`, `*_islem_sure_toplam`

---

## 3. Veri Ön İşleme

### 3.1 Eksik Değer Doldurma (Missing Value Imputation)

**Kod Referansı:** `src/dataset.py:416-428`

```python
# Calculate median for each column (ignoring NaN)
medians = X.median()

# Replace NaN with 0 if median is also NaN
for col in X.columns:
    if pd.isna(medians[col]):
        medians[col] = 0

# Fill missing values
X = X.fillna(medians)
```

**Açıklama:** Eksik değerler ilgili sütunun medyan değeri ile doldurulmuştur. Medyan da NaN ise 0 kullanılmıştır.

### 3.2 Aykırı Değer Sınırlandırma (Outlier Clipping)

**Kod Referansı:** `src/dataset.py:430-436`

```python
# Clip extreme outliers to 99th percentile for each column
for col in X.columns:
    if '_orani' in col or 'islem' in col:  # Ratio columns tend to have outliers
        q99 = X[col].quantile(0.99)
        q01 = X[col].quantile(0.01)
        X[col] = X[col].clip(lower=q01, upper=q99)
```

**Açıklama:** Oran ve işlem sütunlarındaki aykırı değerler %1 ve %99 persentilleri ile sınırlandırılmıştır.

### 3.3 Normalizasyon (StandardScaler)

**Kod Referansı:** `src/dataset.py:507-510`

```python
# Scale features (fit on training data)
X_train_scaled = self.scaler.fit_transform(X_train)
X_val_scaled = self.scaler.transform(X_val)
X_test_scaled = self.scaler.transform(X_test)
```

**Açıklama:** Tüm özellikler `sklearn.preprocessing.StandardScaler` kullanılarak normalize edilmiştir (ortalama=0, standart sapma=1).

### 3.4 Veri Bölme (Train/Validation/Test Split)

**Kod Referansı:** `src/dataset.py:486-494`

```python
# First split: train+val vs test (80/20)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Second split: train vs val (90/10 of remaining)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_size, random_state=random_state
)
```

**Oranlar:**
- Eğitim (Train): %70 (658 örnek)
- Doğrulama (Validation): %10 (74 örnek)
- Test: %20 (184 örnek)

---

## 4. Model Mimarileri

### 4.1 MLP (Multi-Layer Perceptron)

**Kod Referansı:** `src/model.py:18-73`

```
Input(28-35) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
            → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
            → Linear(32)  → BatchNorm → ReLU → Dropout(0.3)
            → Linear(1)   → Output
```

**Parametre Sayısı:** ~14,500-15,500

### 4.2 ResNet (Residual Network)

**Kod Referansı:** `src/model.py:75-163`

```
Input → Linear(128) → BatchNorm → ReLU
      → [ResidualBlock × 3]
      → Linear(1) → Output

ResidualBlock:
    x → Linear → BN → ReLU → Dropout → Linear → BN → (+x) → ReLU
```

**Parametre Sayısı:** ~104,700-105,600

**Skip Connection (Artık Bağlantı):** `src/model.py:103-109`
```python
def forward(self, x):
    residual = x
    out = self.block(x)
    out = out + residual  # Skip connection
    out = self.relu(out)
    return out
```

### 4.3 AttentionNet (Dikkat Mekanizmalı Ağ)

**Kod Referansı:** `src/model.py:165-263`

```
Input → Embedding(128) → Self-Attention → MLP → Output
```

**Self-Attention Mekanizması:** `src/model.py:187-211`
```python
q = self.query(x)
k = self.key(x)
v = self.value(x)

# Attention weights
attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
attn = F.softmax(attn, dim=-1)

# Apply attention to values
out = torch.matmul(attn, v)
```

**Parametre Sayısı:** ~63,800-64,700

---

## 5. Eğitim Süreci

### 5.1 Eğitim Konfigürasyonu

**Kod Referansı:** `train.py:411-430`

| Parametre | Değer | Kod Satırı |
|-----------|-------|------------|
| Kayıp Fonksiyonu | MSE Loss | `train.py:413` |
| Optimizer | Adam | `train.py:417-421` |
| Öğrenme Oranı | 0.001 | `train.py:99` |
| Weight Decay | 1e-4 | `train.py:101-102` |
| Batch Size | 32 | `train.py:97-98` |
| Epoch | 100 | `train.py:95-96` |
| Early Stopping Patience | 15 | `train.py:103-104` |

### 5.2 Öğrenme Oranı Zamanlayıcı

**Kod Referansı:** `train.py:423-430`

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # Kayıp azaldığında iyileşme var
    factor=0.5,      # LR'yi yarıya düşür
    patience=5       # 5 epoch iyileşme olmazsa düşür
)
```

### 5.3 Gradyan Kırpma (Gradient Clipping)

**Kod Referansı:** `train.py:218`

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Açıklama:** Patlayan gradyan problemini önlemek için gradyanlar maksimum 1.0 norm ile sınırlandırılmıştır.

### 5.4 Early Stopping

**Kod Referansı:** `train.py:488-520`

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    # En iyi modeli kaydet
    torch.save({...}, model_path)
else:
    patience_counter += 1
    if patience_counter >= args.patience:
        print(f"Erken durdurma (Early Stopping) - Epoch {epoch+1}")
        break
```

---

## 6. Data Augmentation (Veri Artırma)

### 6.1 Uygulanan Yöntemler

**Kod Referansı:** `src/dataset.py:24-223` (DataAugmenter sınıfı)

#### 6.1.1 Gaussian Noise (Gürültü Ekleme)

**Kod:** `src/dataset.py:46-79`

```python
# Add noise proportional to feature std
noise = np.random.normal(0, self.noise_level, X_selected.shape) * feature_std
X_augmented = X_selected + noise
```

#### 6.1.2 SMOTE-like (Sentetik Örnek Üretimi)

**Kod:** `src/dataset.py:81-123`

```python
# Interpolate between sample and neighbor
alpha = np.random.uniform(0.1, 0.9)
new_sample = sample + alpha * (X[neighbor_idx] - sample)
new_target = target + alpha * (y[neighbor_idx] - target)
```

#### 6.1.3 Mixup Augmentation

**Kod:** `src/dataset.py:125-159`

```python
# Sample mixing coefficient from Beta distribution
lam = np.random.beta(alpha, alpha)

# Mix samples
new_X = lam * X[idx1] + (1 - lam) * X[idx2]
new_y = lam * y[idx1] + (1 - lam) * y[idx2]
```

### 6.2 Data Augmentation Karşılaştırma Sonuçları

**Script:** `compare_augmentation.py`

| Meslek | Baseline R² | En İyi Yöntem | En İyi R² | İyileşme |
|--------|-------------|---------------|-----------|----------|
| Veteriner | 0.7099 | Mixup | 0.7231 | +1.9% |
| Gıda | 0.9150 | SMOTE-like | 0.9163 | +0.1% |
| Ziraat | 0.6850 | SMOTE-like | 0.6916 | +1.0% |

---

## 7. Model Test Sonuçları

### 7.1 Internal Test Sonuçları

**Script:** `test.py --all`

| Meslek | Model | R² | MAE | RMSE |
|--------|-------|-----|-----|------|
| **Veteriner** | MLP | **0.7327** | 1.69 | 2.36 |
| | ResNet | 0.7111 | 1.72 | 2.45 |
| | Attention | 0.7269 | 1.70 | 2.38 |
| **Gıda** | MLP | **0.9156** | 0.38 | 0.67 |
| | ResNet | 0.8965 | 0.41 | 0.75 |
| | Attention | 0.8695 | 0.49 | 0.84 |
| **Ziraat** | MLP | **0.6791** | 2.49 | 3.68 |
| | ResNet | 0.6668 | 2.54 | 3.75 |
| | Attention | 0.6482 | 2.64 | 3.85 |

**En İyi Model:** Her üç meslek için de MLP en iyi performansı göstermiştir.

### 7.2 External Test Sonuçları

**Script:** `external_test.py`

#### Yöntem 1: İl Bazlı (10 il tamamen dışarıda)

**Dışarıda Tutulan İller:** Trabzon, Şanlıurfa, Eskişehir, Muğla, Kayseri, Samsun, Mardin, Çanakkale, Erzurum, Adana

**Kod Referansı:** `external_test.py:55-62`

| Meslek | Internal R² | External R² | Fark |
|--------|-------------|-------------|------|
| Veteriner | 0.6608 | 0.7456 | +0.0848 |
| Gıda | 0.8529 | 0.7641 | -0.0888 |
| Ziraat | 0.7023 | 0.6572 | -0.0451 |

#### Yöntem 2: Hold-out (%20 ilçe tamamen dışarıda)

**Kod Referansı:** `external_test.py:330-390`

| Meslek | Internal R² | External R² | Fark |
|--------|-------------|-------------|------|
| Veteriner | 0.6833 | 0.6644 | -0.0190 |
| Gıda | 0.7979 | 0.8980 | +0.1000 |
| Ziraat | 0.6642 | 0.6641 | -0.0001 |

---

## 8. Değerlendirme Metrikleri

**Kod Referansı:** `test.py:189-235`

### 8.1 MAE (Mean Absolute Error)

```python
mae = np.mean(np.abs(predictions - targets))
```

**Yorum:** Tahmin ile gerçek değer arasındaki ortalama mutlak fark (kişi sayısı cinsinden).

### 8.2 RMSE (Root Mean Squared Error)

```python
mse = np.mean((predictions - targets) ** 2)
rmse = np.sqrt(mse)
```

### 8.3 R² (R-kare / Determinasyon Katsayısı)

```python
ss_res = np.sum((targets - predictions) ** 2)
ss_tot = np.sum((targets - np.mean(targets)) ** 2)
r2 = 1 - (ss_res / ss_tot)
```

**Yorum:** Modelin varyansın yüzde kaçını açıkladığını gösterir. 1'e yakın = iyi.

---

## 9. Dosya Yapısı

```
tarim/
├── train.py                 # Model eğitim scripti (677 satır)
├── test.py                  # Model test scripti (631 satır)
├── compare_augmentation.py  # Data augmentation karşılaştırma (298 satır)
├── external_test.py         # External test seti değerlendirme (554 satır)
├── src/
│   ├── model.py             # Sinir ağı mimarileri (334 satır)
│   └── dataset.py           # Veri yükleme ve ön işleme (588 satır)
├── checkpoints/             # Eğitilmiş model dosyaları (.pt)
├── results/                 # Test sonuçları
└── data/processed/          # İşlenmiş veri dosyaları
```

---

## 10. Kullanım

### 10.1 Tüm Modelleri Eğitme

```bash
python train.py --all
```

### 10.2 Tüm Modelleri Test Etme

```bash
python test.py --all
```

### 10.3 Data Augmentation Karşılaştırması

```bash
python compare_augmentation.py
```

### 10.4 External Test

```bash
python external_test.py
```

---

## 11. Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| PyTorch | 2.0+ | Derin öğrenme framework |
| pandas | - | Veri manipülasyonu |
| numpy | - | Sayısal hesaplamalar |
| scikit-learn | - | StandardScaler, train_test_split |

---

## 12. Sonuç

- **En iyi model:** MLP (tüm meslekler için)
- **En yüksek R²:** Gıda Mühendisi (0.9156)
- **Data augmentation etkisi:** Marjinal iyileşme (~1-2%)
- **External test:** Model genelleme yeteneği doğrulandı

---

*Bu rapor, proje kodlarından otomatik olarak çıkarılmış referanslar içermektedir.*
