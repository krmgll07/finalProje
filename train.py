#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TARIM PERSONELİ NORM TAHMİN MODELİ - EĞİTİM SCRIPTI
================================================================================
Bu script, Türkiye'deki tarım ilçeleri için optimal personel tahsisini tahmin
eden derin öğrenme modellerini eğitir.

AMAÇ:
    Veteriner Hekim, Gıda Mühendisi ve Ziraat Mühendisi personel ihtiyacını
    ilçelerin demografik ve tarımsal verilerine göre tahmin etmek.

KULLANIM:
    python train.py --profession veteriner --model mlp --epochs 100
    python train.py --profession gida --model resnet --epochs 150
    python train.py --profession ziraat --model attention --epochs 100

MODEL MİMARİLERİ:
    - MLP (Multi-Layer Perceptron): Çok katmanlı tam bağlantılı ağ
    - ResNet: Artık bağlantılı (residual) bloklar içeren ağ
    - Attention: Dikkat mekanizması kullanan ağ

Yazar: Kerim GÜLLÜ
Tarih: Aralık 2024
Ders: AIE521 - Derin Öğrenme
================================================================================
"""

# =============================================================================
# KÜTÜPHANE İMPORTLARI
# =============================================================================
import argparse          # Komut satırı argümanlarını işlemek için
import os                # Dosya/dizin işlemleri için
import json              # JSON formatında veri kaydetmek için
import time              # Eğitim süresini ölçmek için
from datetime import datetime  # Zaman damgası oluşturmak için

import numpy as np       # Sayısal hesaplamalar için temel kütüphane
import torch             # PyTorch derin öğrenme framework'ü
import torch.nn as nn    # Sinir ağı modülleri (kayıp fonksiyonları vb.)
import torch.optim as optim  # Optimizasyon algoritmaları (Adam, SGD vb.)
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Öğrenme oranı zamanlayıcı

# Proje modüllerini import et
# src klasörünü Python path'ine ekleyerek kendi yazdığımız modülleri kullanabiliyoruz
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import DataPreprocessor  # Veri ön işleme sınıfı
from model import get_model, count_parameters  # Model oluşturma fonksiyonları


# =============================================================================
# KOMUT SATIRI ARGÜMANLARI
# =============================================================================
def parse_args():
    """
    Komut satırı argümanlarını ayrıştır ve döndür.

    Bu fonksiyon, kullanıcının terminal üzerinden script'e parametre
    geçirmesini sağlar. Örneğin:
        python train.py --profession gida --epochs 200

    Returns:
        argparse.Namespace: Tüm argümanları içeren nesne
    """
    parser = argparse.ArgumentParser(
        description='Personel Norm Tahmin Modeli Eğitimi'
    )

    # -------------------------------------------------------------------------
    # VERİ ARGÜMANLARI
    # -------------------------------------------------------------------------
    parser.add_argument('--profession', type=str, default='veteriner',
                        choices=['veteriner', 'gida', 'ziraat'],
                        help='Eğitilecek meslek türü: veteriner, gida veya ziraat')
    parser.add_argument('--all', action='store_true',
                        help='Tüm meslekler ve tüm modeller için eğitim yap')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Veri dosyalarının bulunduğu dizin')

    # -------------------------------------------------------------------------
    # MODEL ARGÜMANLARI
    # -------------------------------------------------------------------------
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'resnet', 'attention'],
                        help='Kullanılacak model mimarisi')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Gizli katman boyutu (nöron sayısı)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout oranı (aşırı öğrenmeyi önlemek için)')

    # -------------------------------------------------------------------------
    # EĞİTİM ARGÜMANLARI
    # -------------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=100,
                        help='Eğitim döngüsü sayısı (epoch)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini-batch boyutu (her adımda işlenen örnek sayısı)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Öğrenme oranı (learning rate)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 regularizasyon katsayısı (ağırlık çürümesi)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping için sabır parametresi')

    # -------------------------------------------------------------------------
    # ÇIKTI ARGÜMANLARI
    # -------------------------------------------------------------------------
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Model checkpoint\'lerinin kaydedileceği dizin')
    parser.add_argument('--seed', type=int, default=42,
                        help='Rastgelelik tohumu (reproducibility için)')

    # -------------------------------------------------------------------------
    # VERİ ARTIRMA (DATA AUGMENTATION) ARGÜMANLARI
    # -------------------------------------------------------------------------
    parser.add_argument('--augment', action='store_true',
                        help='Eğitim verisine veri artırma uygula')
    parser.add_argument('--augment_factor', type=float, default=1.0,
                        help='Artırma faktörü (1.0 = veriyi iki katına çıkar)')
    parser.add_argument('--augment_methods', type=str, nargs='+',
                        default=['noise', 'smote', 'mixup'],
                        choices=['noise', 'smote', 'mixup', 'jitter'],
                        help='Kullanılacak artırma yöntemleri')

    return parser.parse_args()


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
def set_seed(seed: int):
    """
    Rastgelelik tohumunu ayarla - Tekrarlanabilirlik için kritik!

    Derin öğrenme modelleri rastgele başlatılır. Aynı sonuçları elde etmek
    için tüm rastgelelik kaynaklarının sabitlenmesi gerekir.

    Args:
        seed: Rastgelelik tohumu (örn: 42)

    Ayarlanan rastgelelik kaynakları:
        - NumPy random
        - PyTorch CPU random
        - PyTorch CUDA random (GPU varsa)
        - cuDNN deterministik mod
    """
    np.random.seed(seed)          # NumPy için
    torch.manual_seed(seed)        # PyTorch CPU için

    # GPU (CUDA) varsa onun da rastgeleliğini sabitle
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Çoklu GPU için
        # cuDNN'i deterministik moda al (biraz yavaşlatır ama tekrarlanabilir)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# EĞİTİM FONKSİYONLARI
# =============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Modeli bir epoch (tam veri seti geçişi) boyunca eğit.

    Bir epoch, tüm eğitim verisinin bir kez modelden geçirilmesi demektir.
    Bu fonksiyon mini-batch'ler halinde veriyi işler.

    Args:
        model: Eğitilecek PyTorch modeli
        train_loader: Eğitim verisi DataLoader'ı
        criterion: Kayıp fonksiyonu (MSE Loss kullanıyoruz)
        optimizer: Optimizasyon algoritması (Adam)
        device: İşlem yapılacak cihaz (CPU veya CUDA)

    Returns:
        float: Bu epoch'taki ortalama eğitim kaybı

    EĞİTİM ADIMLARI:
        1. İleri geçiş (forward pass): Girdiyi modelden geçir
        2. Kayıp hesapla: Tahmin ile gerçek değeri karşılaştır
        3. Geri yayılım (backward pass): Gradyanları hesapla
        4. Ağırlık güncelle: Optimizer ile parametreleri güncelle
    """
    # Modeli eğitim moduna al
    # Bu, dropout ve batch normalization'ın eğitim davranışını aktifleştirir
    model.train()

    total_loss = 0.0  # Toplam kayıp
    num_batches = 0   # Batch sayacı

    # Tüm mini-batch'leri işle
    for features, targets in train_loader:
        # Veriyi uygun cihaza taşı (GPU varsa GPU'ya)
        features = features.to(device)
        targets = targets.to(device)

        # ADIM 1: Gradyanları sıfırla
        # Her batch'te gradyanlar sıfırlanmalı, aksi halde birikir
        optimizer.zero_grad()

        # ADIM 2: İleri geçiş (Forward Pass)
        # Modelden tahmin al
        outputs = model(features)

        # ADIM 3: Kayıp hesapla
        # MSE = Mean Squared Error = Ortalama Kare Hata
        loss = criterion(outputs, targets)

        # ADIM 4: Geri yayılım (Backward Pass)
        # Zincir kuralı ile tüm gradyanları hesapla
        loss.backward()

        # ADIM 5: Gradyan kırpma (Gradient Clipping)
        # Patlayan gradyan problemini önlemek için gradyanları sınırla
        # max_norm=1.0: Gradyan vektörünün normu 1'i aşamaz
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ADIM 6: Ağırlıkları güncelle
        # w = w - lr * gradient formülü ile güncelleme
        optimizer.step()

        # İstatistikleri güncelle
        total_loss += loss.item()
        num_batches += 1

    # Ortalama kaybı döndür
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """
    Modeli doğrulama (validation) seti üzerinde değerlendir.

    Doğrulama seti, modelin eğitim sırasında görmediği veri üzerindeki
    performansını ölçmek için kullanılır. Aşırı öğrenmeyi (overfitting)
    tespit etmek için kritiktir.

    Args:
        model: Değerlendirilecek PyTorch modeli
        val_loader: Doğrulama verisi DataLoader'ı
        criterion: Kayıp fonksiyonu
        device: İşlem yapılacak cihaz

    Returns:
        tuple: (ortalama_kayıp, MAE, R²_skoru)

    METRİKLER:
        - Loss: MSE kaybı (eğitim hedefi)
        - MAE: Mean Absolute Error - Ortalama mutlak hata (kişi sayısı cinsinden)
        - R²: Determinasyon katsayısı (0-1 arası, 1'e yakın = iyi)
    """
    # Modeli değerlendirme moduna al
    # Dropout devre dışı kalır, batch norm sabit istatistikler kullanır
    model.eval()

    total_loss = 0.0
    all_predictions = []  # Tüm tahminleri topla
    all_targets = []      # Tüm gerçek değerleri topla

    # Gradyan hesaplama kapalı - sadece ileri geçiş yapıyoruz
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)

            # Tahmin al
            outputs = model(features)
            loss = criterion(outputs, targets)

            # Sonuçları topla
            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # NumPy array'e çevir
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # -------------------------------------------------------------------------
    # METRİK HESAPLAMALARI
    # -------------------------------------------------------------------------

    # MAE: Mean Absolute Error (Ortalama Mutlak Hata)
    # Tahmin ile gerçek arasındaki ortalama fark (kişi sayısı)
    mae = np.mean(np.abs(predictions - targets))

    # R² (R-kare) Skoru: Determinasyon Katsayısı
    # Modelin varyansın ne kadarını açıkladığını gösterir
    # Formül: R² = 1 - (SS_res / SS_tot)
    # SS_res: Residual (kalan) kareler toplamı
    # SS_tot: Toplam kareler toplamı
    ss_res = np.sum((targets - predictions) ** 2)  # Hata kareler toplamı
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)  # Toplam varyans
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Ortalama kayıp
    avg_loss = total_loss / len(val_loader)

    return avg_loss, mae, r2


# =============================================================================
# ANA EĞİTİM FONKSİYONU
# =============================================================================
def train(args):
    """
    Ana eğitim fonksiyonu - Tüm eğitim sürecini yönetir.

    Bu fonksiyon:
        1. Veriyi yükler ve hazırlar
        2. Modeli oluşturur
        3. Eğitim döngüsünü çalıştırır
        4. Early stopping uygular
        5. En iyi modeli kaydeder
        6. Sonuçları raporlar

    Args:
        args: Komut satırı argümanları (profession, model, epochs, vb.)

    Returns:
        tuple: (eğitilmiş_model, eğitim_geçmişi, sonuçlar)
    """
    # =========================================================================
    # BAŞLANGIÇ BİLGİLERİ
    # =========================================================================
    print("="*60)
    print("PERSONEL NORM TAHMİN - EĞİTİM")
    print("="*60)
    print(f"\nMeslek: {args.profession.upper()}")
    print(f"Model: {args.model.upper()}")
    print(f"Cihaz: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")

    # Rastgelelik tohumunu ayarla
    set_seed(args.seed)

    # =========================================================================
    # CİHAZ AYARLARI
    # =========================================================================
    # CUDA (GPU) kullanılabilirliğini kontrol et
    try:
        if torch.cuda.is_available():
            # GPU'nun gerçekten çalıştığını test et
            torch.cuda.get_device_name(0)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    except:
        # CUDA hatası olursa CPU'ya düş
        device = torch.device('cpu')

    # Model kayıt dizinini oluştur
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # VERİ YÜKLEME
    # =========================================================================
    print("\n" + "-"*40)
    print("Veri yükleniyor...")
    print("-"*40)

    # DataPreprocessor: Veriyi yükler, temizler, normalize eder
    preprocessor = DataPreprocessor(args.profession, args.data_dir)

    # DataLoader'ları oluştur
    # train_loader: Eğitim verisi (%70)
    # val_loader: Doğrulama verisi (%10)
    # test_loader: Test verisi (%20)
    # input_dim: Girdi özellik sayısı
    train_loader, val_loader, test_loader, input_dim = preprocessor.create_dataloaders(
        batch_size=args.batch_size,
        augment=args.augment,
        augment_factor=args.augment_factor,
        augment_methods=args.augment_methods
    )

    # =========================================================================
    # MODEL OLUŞTURMA
    # =========================================================================
    print("\n" + "-"*40)
    print("Model oluşturuluyor...")
    print("-"*40)

    # Model hiperparametrelerini ayarla
    model_kwargs = {
        'hidden_dim': args.hidden_dim,
        'dropout_rate': args.dropout
    }

    # MLP için özel katman yapılandırması
    if args.model == 'mlp':
        model_kwargs = {
            # Azalan katman boyutları: 128 -> 64 -> 32
            'hidden_dims': [args.hidden_dim, args.hidden_dim // 2, args.hidden_dim // 4],
            'dropout_rate': args.dropout
        }

    # Modeli oluştur ve cihaza taşı
    model = get_model(args.model, input_dim, **model_kwargs)
    model = model.to(device)

    print(f"Model mimarisi: {args.model}")
    print(f"Girdi boyutu: {input_dim}")
    print(f"Toplam parametre sayısı: {count_parameters(model):,}")

    # =========================================================================
    # KAYIP FONKSİYONU VE OPTİMİZER
    # =========================================================================

    # MSE Loss: Regresyon problemleri için standart kayıp fonksiyonu
    # L = (1/n) * Σ(y_pred - y_true)²
    criterion = nn.MSELoss()

    # Adam Optimizer: Adaptif öğrenme oranlı optimizer
    # Momentum ve RMSprop'un avantajlarını birleştirir
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,                    # Öğrenme oranı
        weight_decay=args.weight_decay  # L2 regularizasyon
    )

    # ReduceLROnPlateau: Öğrenme oranı zamanlayıcısı
    # Validation loss iyileşmezse öğrenme oranını düşürür
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      # Kayıp azaldığında iyileşme var
        factor=0.5,      # LR'yi yarıya düşür
        patience=5       # 5 epoch iyileşme olmazsa düşür
    )

    # =========================================================================
    # EĞİTİM DÖNGÜSÜ
    # =========================================================================
    print("\n" + "-"*40)
    print("Eğitim başlıyor...")
    print("-"*40)

    # En iyi model takibi için değişkenler
    best_val_loss = float('inf')  # En düşük validation kaybı
    best_val_r2 = 0               # En yüksek R² skoru
    patience_counter = 0           # Early stopping sayacı

    # Eğitim geçmişi - grafik çizmek için
    history = {
        'train_loss': [],    # Eğitim kaybı
        'val_loss': [],      # Doğrulama kaybı
        'val_mae': [],       # Doğrulama MAE
        'val_r2': [],        # Doğrulama R²
        'lr': []             # Öğrenme oranı
    }

    start_time = time.time()  # Eğitim süresini ölç

    # -------------------------------------------------------------------------
    # ANA EĞİTİM DÖNGÜSÜ
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # Bir epoch eğitim yap
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Doğrulama setinde değerlendir
        val_loss, val_mae, val_r2 = validate(model, val_loader, criterion, device)

        # Öğrenme oranını güncelle (gerekirse düşür)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Geçmişe kaydet
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        # Her 10 epoch'ta bir veya ilk epoch'ta ilerlemeyi yazdır
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} | "
                  f"Val R2: {val_r2:.4f} | "
                  f"LR: {current_lr:.6f}")

        # ---------------------------------------------------------------------
        # EN İYİ MODEL KONTROLÜ VE EARLY STOPPING
        # ---------------------------------------------------------------------
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
                'epoch': epoch,                              # Hangi epoch
                'model_state_dict': model.state_dict(),      # Model ağırlıkları
                'optimizer_state_dict': optimizer.state_dict(),  # Optimizer durumu
                'val_loss': val_loss,                        # Validation kaybı
                'val_mae': val_mae,                          # Validation MAE
                'val_r2': val_r2,                            # Validation R²
                'input_dim': input_dim,                      # Girdi boyutu
                'model_type': args.model,                    # Model türü
                'profession': args.profession,               # Meslek
                'scaler': preprocessor.get_scaler(),         # Normalizasyon scaler'ı
                'feature_names': preprocessor.get_feature_names()  # Özellik isimleri
            }, model_path)
        else:
            # İyileşme yok, early stopping sayacını artır
            patience_counter += 1
            if patience_counter >= args.patience:
                # Patience aşıldı, eğitimi erken durdur
                print(f"\nErken durdurma (Early Stopping) - Epoch {epoch+1}")
                break

    training_time = time.time() - start_time  # Toplam eğitim süresi

    # =========================================================================
    # FİNAL DEĞERLENDİRME (TEST SETİ)
    # =========================================================================
    print("\n" + "-"*40)
    print("Test Seti Üzerinde Final Değerlendirme")
    print("-"*40)

    # En iyi modeli yükle
    model_path = os.path.join(args.save_dir, f'{args.profession}_{args.model}_best.pt')
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test setinde değerlendir
    test_loss, test_mae, test_r2 = validate(model, test_loader, criterion, device)

    print(f"Test Kaybı (Loss): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} kişi")
    print(f"Test R²: {test_r2:.4f}")

    # =========================================================================
    # SONUÇLARI KAYDET
    # =========================================================================

    # Eğitim geçmişini JSON olarak kaydet
    history_path = os.path.join(args.save_dir, f'{args.profession}_{args.model}_history.json')
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # Eğitim sonuçlarını JSON olarak kaydet
    results = {
        'profession': args.profession,
        'model': args.model,
        'input_dim': int(input_dim),
        'epochs_trained': int(epoch + 1),
        'training_time_seconds': float(training_time),
        'best_val_loss': float(best_val_loss),
        'best_val_r2': float(best_val_r2),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_r2': float(test_r2),
        'parameters': int(count_parameters(model)),
        'timestamp': datetime.now().isoformat(),
        'config': vars(args)
    }

    results_path = os.path.join(args.save_dir, f'{args.profession}_{args.model}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # =========================================================================
    # ÖZET RAPOR
    # =========================================================================
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI")
    print("="*60)
    print(f"En İyi Validation R²: {best_val_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f} personel")
    print(f"Eğitim süresi: {training_time/60:.1f} dakika")
    print(f"\nModel kaydedildi: {model_path}")
    print(f"Geçmiş kaydedildi: {history_path}")
    print(f"Sonuçlar kaydedildi: {results_path}")

    return model, history, results


# =============================================================================
# TÜM MESLEKLERİ VE MODELLERİ EĞİT
# =============================================================================
def train_all(args):
    """
    Tüm meslekler ve tüm modeller için eğitim yap.

    Bu fonksiyon:
        - 3 Meslek: Veteriner, Gıda Mühendisi, Ziraat Mühendisi
        - 3 Model: MLP, ResNet, Attention

    Toplam 9 model eğitir ve sonuçları karşılaştırır.

    Args:
        args: Komut satırı argümanları
    """
    professions = ['veteriner', 'gida', 'ziraat']
    models = ['mlp', 'resnet', 'attention']
    all_results = {}

    total = len(professions) * len(models)
    current = 0

    for prof in professions:
        all_results[prof] = {}
        for model_type in models:
            current += 1
            print("\n" + "#"*70)
            print(f"# [{current}/{total}] {prof.upper()} - {model_type.upper()} MODELİ EĞİTİLİYOR")
            print("#"*70)

            args.profession = prof
            args.model = model_type
            _, _, results = train(args)
            all_results[prof][model_type] = results

    # Sonuçları karşılaştır
    print("\n" + "="*80)
    print("TÜM MODELLERİN KARŞILAŞTIRMASI")
    print("="*80)
    print(f"{'Meslek':<12} | {'Model':<10} | {'Test R²':>10} | {'Test MAE':>10} | {'Parametre':>12}")
    print("-"*80)
    for prof in professions:
        for model_type in models:
            res = all_results[prof][model_type]
            print(f"{prof:<12} | {model_type:<10} | {res['test_r2']:>10.4f} | {res['test_mae']:>10.2f} | {res['parameters']:>12,}")

    # En iyi modelleri göster
    print("\n" + "="*80)
    print("HER MESLEK İÇİN EN İYİ MODEL")
    print("="*80)
    for prof in professions:
        best_model = max(all_results[prof].items(), key=lambda x: x[1]['test_r2'])
        print(f"{prof.upper()}: {best_model[0].upper()} (R²={best_model[1]['test_r2']:.4f})")

    return all_results


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================
if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında bu blok çalışır.

    Kullanım örnekleri:
        # Tüm meslekler ve modeller:
        python train.py --all

        # Tek meslek eğitimi:
        python train.py --profession veteriner --model mlp --epochs 100

        # Farklı model:
        python train.py --profession gida --model resnet --epochs 150

        # Veri artırma ile:
        python train.py --profession ziraat --augment --epochs 100
    """
    args = parse_args()

    # --all flag'i varsa tüm meslekleri ve modelleri eğit
    if args.all:
        train_all(args)
    else:
        train(args)
