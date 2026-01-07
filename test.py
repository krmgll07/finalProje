#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TARIM PERSONELİ NORM TAHMİN MODELİ - TEST SCRIPTI
================================================================================
Bu script, eğitilmiş derin öğrenme modellerini yükler ve test verisi üzerinde
değerlendirerek performans metriklerini hesaplar.

AMAÇ:
    Eğitim tamamlandıktan sonra modelin gerçek performansını ölçmek.
    Test seti, model tarafından hiç görülmemiş veridir.

KULLANIM:
    python test.py --profession veteriner --model mlp
    python test.py --profession gida --model resnet --visualize
    python test.py --all

ÇIKTILAR:
    - Performans metrikleri (MAE, RMSE, R², MAPE)
    - Sınıflandırma sonuçları (fazla/eksik/dengede)
    - Görselleştirme grafikleri (opsiyonel)

Yazar: Kerim GÜLLÜ
Tarih: Aralık 2024
Ders: AIE521 - Derin Öğrenme
================================================================================
"""

# =============================================================================
# KÜTÜPHANE İMPORTLARI
# =============================================================================
import argparse          # Komut satırı argümanları için
import os                # Dosya/dizin işlemleri için
import json              # JSON okuma/yazma için

import numpy as np       # Sayısal hesaplamalar için
import pandas as pd      # Veri manipülasyonu için
import torch             # PyTorch framework'ü
import matplotlib.pyplot as plt  # Grafik çizimi için

# Proje modüllerini import et
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import DataPreprocessor  # Veri ön işleme
from model import get_model, count_parameters  # Model fonksiyonları


# =============================================================================
# KOMUT SATIRI ARGÜMANLARI
# =============================================================================
def parse_args():
    """
    Komut satırı argümanlarını ayrıştır.

    Kullanım örnekleri:
        python test.py --profession veteriner
        python test.py --all --visualize

    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar
    """
    parser = argparse.ArgumentParser(
        description='Personel Norm Tahmin Modeli Testi'
    )

    # Test edilecek meslek
    parser.add_argument('--profession', type=str, default='veteriner',
                        choices=['veteriner', 'gida', 'ziraat', 'all'],
                        help='Test edilecek meslek türü')

    # Kullanılacak model mimarisi
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'resnet', 'attention'],
                        help='Model mimarisi')

    # Dizin ayarları
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Veri dosyalarının bulunduğu dizin')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Model checkpoint dosyalarının dizini')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Sonuçların kaydedileceği dizin')

    # Opsiyonel ayarlar
    parser.add_argument('--visualize', action='store_true',
                        help='Görselleştirme grafikleri oluştur')
    parser.add_argument('--all', action='store_true',
                        help='Tüm meslekleri test et')

    return parser.parse_args()


# =============================================================================
# MODEL YÜKLEME
# =============================================================================
def load_model(checkpoint_path: str, device: torch.device):
    """
    Eğitilmiş modeli checkpoint dosyasından yükle.

    Checkpoint dosyası şunları içerir:
        - model_state_dict: Model ağırlıkları
        - input_dim: Girdi boyutu
        - model_type: Model türü (mlp/resnet/attention)
        - scaler: Normalizasyon için kullanılan scaler
        - feature_names: Özellik isimleri

    Args:
        checkpoint_path: .pt dosyasının yolu
        device: Modelin yükleneceği cihaz (CPU/CUDA)

    Returns:
        tuple: (model, checkpoint_verisi)
    """
    # Checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Model parametrelerini al
    input_dim = checkpoint['input_dim']    # Girdi özellik sayısı
    model_type = checkpoint['model_type']  # Model türü

    # Model mimarisine göre hiperparametreleri ayarla
    if model_type == 'mlp':
        # MLP için katman boyutları
        model_kwargs = {'hidden_dims': [128, 64, 32], 'dropout_rate': 0.3}
    else:
        # ResNet ve Attention için
        model_kwargs = {'hidden_dim': 128, 'dropout_rate': 0.3}

    # Modeli oluştur
    model = get_model(model_type, input_dim, **model_kwargs)

    # Kaydedilmiş ağırlıkları yükle
    model.load_state_dict(checkpoint['model_state_dict'])

    # Modeli cihaza taşı ve değerlendirme moduna al
    model = model.to(device)
    model.eval()  # Dropout kapalı, batch norm sabit

    return model, checkpoint


# =============================================================================
# MODEL DEĞERLENDİRME
# =============================================================================
def evaluate_model(model, test_loader, device):
    """
    Modeli test verisi üzerinde değerlendir ve metrikleri hesapla.

    HESAPLANAN METRİKLER:
        - MAE (Mean Absolute Error): Ortalama mutlak hata
        - MSE (Mean Squared Error): Ortalama kare hata
        - RMSE (Root MSE): Kök ortalama kare hata
        - R² (R-kare): Determinasyon katsayısı
        - MAPE (Mean Absolute Percentage Error): Ortalama mutlak yüzde hata

    Args:
        model: Değerlendirilecek PyTorch modeli
        test_loader: Test verisi DataLoader'ı
        device: İşlem cihazı

    Returns:
        dict: Tahminler, hedefler ve metrikler içeren sözlük
    """
    # Değerlendirme moduna al (dropout kapalı)
    model.eval()

    all_predictions = []  # Tüm tahminleri topla
    all_targets = []      # Tüm gerçek değerleri topla

    # Gradyan hesaplama kapalı - sadece tahmin yapıyoruz
    with torch.no_grad():
        for features, targets in test_loader:
            # Veriyi cihaza taşı
            features = features.to(device)

            # Tahmin al
            outputs = model(features)

            # Sonuçları listeye ekle
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # NumPy array'e çevir
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # =========================================================================
    # METRİK HESAPLAMALARI
    # =========================================================================

    # MAE: Mean Absolute Error (Ortalama Mutlak Hata)
    # Formül: MAE = (1/n) * Σ|y_pred - y_true|
    # Yorum: Ortalama kaç kişi hata yapıyoruz?
    mae = np.mean(np.abs(predictions - targets))

    # MSE: Mean Squared Error (Ortalama Kare Hata)
    # Formül: MSE = (1/n) * Σ(y_pred - y_true)²
    # Büyük hataları daha fazla cezalandırır
    mse = np.mean((predictions - targets) ** 2)

    # RMSE: Root Mean Squared Error (Kök Ortalama Kare Hata)
    # Formül: RMSE = √MSE
    # MAE ile aynı birimde (kişi sayısı)
    rmse = np.sqrt(mse)

    # R²: R-kare (Determinasyon Katsayısı)
    # Formül: R² = 1 - (SS_res / SS_tot)
    # Yorum: Modelin varyansın yüzde kaçını açıkladığı
    # 0-1 arası, 1'e yakın = iyi
    ss_res = np.sum((targets - predictions) ** 2)  # Kalan kareler toplamı
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)  # Toplam kareler toplamı
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # MAPE: Mean Absolute Percentage Error (Ortalama Mutlak Yüzde Hata)
    # Formül: MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100
    # Yorum: Yüzde olarak ortalama hata
    # NOT: Sıfır değerler için hesaplanamaz
    non_zero_mask = targets != 0  # Sıfır olmayan hedefler
    if non_zero_mask.any():
        mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask])
                              / targets[non_zero_mask])) * 100
    else:
        mape = 0

    return {
        'predictions': predictions,  # Model tahminleri
        'targets': targets,          # Gerçek değerler
        'mae': mae,                  # Ortalama mutlak hata
        'mse': mse,                  # Ortalama kare hata
        'rmse': rmse,                # Kök ortalama kare hata
        'r2': r2,                    # R-kare skoru
        'mape': mape                 # Yüzde hata
    }


# =============================================================================
# TAHMİN SINIFLANDIRMA
# =============================================================================
def classify_predictions(predictions: np.ndarray, targets: np.ndarray,
                         threshold: float = 0.2):
    """
    Tahminleri norm fazlası, norm eksiği veya dengede olarak sınıflandır.

    SINIFLANDIRMA KURALLARI:
        - Göreceli fark < threshold (20%): DENGEDE
        - Gerçek > Tahmin: NORM FAZLASI (fazla personel var)
        - Gerçek < Tahmin: NORM EKSİĞİ (personel eksik)

    Args:
        predictions: Model tahminleri (ideal norm)
        targets: Gerçek personel sayıları
        threshold: Dengede sayılması için eşik değer (varsayılan %20)

    Returns:
        dict: Sınıflandırma sonuçları ve sayıları
    """
    # Farkları hesapla: pozitif = fazla, negatif = eksik
    differences = targets - predictions

    # Göreceli fark hesapla (sıfıra bölmeyi önle)
    relative_diff = np.abs(differences) / np.maximum(predictions, 1)

    # Her ilçe için sınıflandırma yap
    classifications = []
    for diff, rel_diff in zip(differences, relative_diff):
        if rel_diff < threshold:
            # Göreceli fark eşik altında = dengede
            classifications.append('dengede')
        elif diff > 0:
            # Gerçek > Tahmin = fazla personel var
            classifications.append('norm_fazlasi')
        else:
            # Gerçek < Tahmin = personel eksik
            classifications.append('norm_eksigi')

    classifications = np.array(classifications)

    # Her kategorideki ilçe sayısını hesapla
    counts = {
        'norm_fazlasi': np.sum(classifications == 'norm_fazlasi'),
        'norm_eksigi': np.sum(classifications == 'norm_eksigi'),
        'dengede': np.sum(classifications == 'dengede')
    }

    return {
        'classifications': classifications,  # Her ilçenin durumu
        'counts': counts,                    # Kategori sayıları
        'differences': differences           # Fark değerleri
    }


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================
def visualize_results(results: dict, profession: str, output_dir: str):
    """
    Model sonuçlarını görselleştiren grafikler oluştur.

    OLUŞTURULAN GRAFİKLER:
        1. Scatter Plot: Tahmin vs Gerçek değerler
        2. Residual Plot: Hata dağılımı
        3. Histogram: Hataların frekans dağılımı
        4. Pie Chart: Durum sınıflandırması oranları

    Args:
        results: evaluate_model fonksiyonundan gelen sonuçlar
        profession: Meslek adı (grafik başlığı için)
        output_dir: Grafiklerin kaydedileceği dizin
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)

    predictions = results['predictions']
    targets = results['targets']

    # 2x2 subplot oluştur
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # =========================================================================
    # GRAFİK 1: SCATTER PLOT - TAHMİN VS GERÇEK
    # =========================================================================
    ax1 = axes[0, 0]
    ax1.scatter(targets, predictions, alpha=0.5, edgecolors='k', linewidth=0.5)

    # Mükemmel tahmin çizgisi (y = x)
    max_val = max(targets.max(), predictions.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Mükemmel Tahmin')

    ax1.set_xlabel('Gerçek Personel Sayısı')
    ax1.set_ylabel('Tahmin Edilen Personel Sayısı')
    ax1.set_title(f'{profession.upper()} - Tahmin vs Gerçek')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # GRAFİK 2: RESİDUAL PLOT - HATA DAĞILIMI
    # =========================================================================
    ax2 = axes[0, 1]
    residuals = targets - predictions  # Hatalar

    ax2.scatter(predictions, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')  # Sıfır hata çizgisi

    ax2.set_xlabel('Tahmin Edilen Personel Sayısı')
    ax2.set_ylabel('Hata (Gerçek - Tahmin)')
    ax2.set_title('Residual (Hata) Grafiği')
    ax2.grid(True, alpha=0.3)

    # =========================================================================
    # GRAFİK 3: HİSTOGRAM - HATA FREKANSI
    # =========================================================================
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)

    # Referans çizgileri
    ax3.axvline(x=0, color='r', linestyle='--', label='Sıfır Hata')
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='--',
                label=f'Ortalama: {np.mean(residuals):.2f}')

    ax3.set_xlabel('Hata Değeri')
    ax3.set_ylabel('Frekans')
    ax3.set_title('Hata Dağılımı Histogramı')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # =========================================================================
    # GRAFİK 4: PIE CHART - SINIFLANDIRMA ORANLARI
    # =========================================================================
    ax4 = axes[1, 1]
    class_results = classify_predictions(predictions, targets)
    counts = class_results['counts']

    labels = ['Fazla (Norm Fazlası)', 'Eksik (Norm Eksiği)', 'Dengede']
    sizes = [counts['norm_fazlasi'], counts['norm_eksigi'], counts['dengede']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']  # Yeşil, Kırmızı, Turuncu
    explode = (0.05, 0.05, 0.05)  # Dilimler hafif ayrık

    ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax4.set_title('Personel Durumu Dağılımı')

    # Grafikleri kaydet
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{profession}_evaluation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Görselleştirme kaydedildi: {output_dir}/{profession}_evaluation.png")


# =============================================================================
# TEK MESLEK TESTİ
# =============================================================================
def test_profession(profession: str, model_type: str, args):
    """
    Belirli bir meslek için eğitilmiş modeli test et.

    TEST SÜRECİ:
        1. Model checkpoint'ini yükle
        2. Test verisini hazırla
        3. Tahminler yap
        4. Metrikleri hesapla
        5. Sonuçları kaydet

    Args:
        profession: Test edilecek meslek (veteriner/gida/ziraat)
        model_type: Model mimarisi (mlp/resnet/attention)
        args: Komut satırı argümanları

    Returns:
        dict: Test sonuçları veya None (hata durumunda)
    """
    print("\n" + "="*60)
    print(f"TEST: {profession.upper()} - {model_type.upper()}")
    print("="*60)

    # Cihaz seçimi (GPU varsa GPU, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")

    # =========================================================================
    # MODEL YÜKLEME
    # =========================================================================
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'{profession}_{model_type}_best.pt'
    )

    # Checkpoint dosyası var mı kontrol et
    if not os.path.exists(checkpoint_path):
        print(f"HATA: Checkpoint bulunamadı: {checkpoint_path}")
        print("Lütfen önce train.py ile modeli eğitin.")
        return None

    print(f"Model yükleniyor: {checkpoint_path}")
    model, checkpoint = load_model(checkpoint_path, device)

    # =========================================================================
    # TEST VERİSİ YÜKLEME
    # =========================================================================
    print("Test verisi yükleniyor...")
    preprocessor = DataPreprocessor(profession, args.data_dir)

    # Eğitimde kullanılan scaler'ı kullan (tutarlılık için)
    preprocessor.scaler = checkpoint['scaler']

    # DataLoader'ları oluştur (sadece test_loader kullanılacak)
    _, _, test_loader, input_dim = preprocessor.create_dataloaders(
        batch_size=32
    )

    # =========================================================================
    # DEĞERLENDİRME
    # =========================================================================
    print("Model değerlendiriliyor...")
    results = evaluate_model(model, test_loader, device)

    # Sınıflandırma yap
    class_results = classify_predictions(results['predictions'], results['targets'])
    results['classification'] = class_results

    # =========================================================================
    # SONUÇLARI YAZDIR
    # =========================================================================
    print("\n" + "-"*40)
    print("TEST SONUÇLARI")
    print("-"*40)
    print(f"Ortalama Mutlak Hata (MAE): {results['mae']:.2f} personel")
    print(f"Kök Ortalama Kare Hata (RMSE): {results['rmse']:.2f}")
    print(f"R-kare (R²): {results['r2']:.4f}")
    print(f"Ortalama Yüzde Hata (MAPE): {results['mape']:.1f}%")

    print("\n" + "-"*40)
    print("SINIFLANDIRMA SONUÇLARI")
    print("-"*40)
    counts = class_results['counts']
    total = sum(counts.values())
    print(f"Norm Fazlası (Fazla Personel): {counts['norm_fazlasi']} ({100*counts['norm_fazlasi']/total:.1f}%)")
    print(f"Norm Eksiği (Eksik Personel): {counts['norm_eksigi']} ({100*counts['norm_eksigi']/total:.1f}%)")
    print(f"Dengede: {counts['dengede']} ({100*counts['dengede']/total:.1f}%)")

    # =========================================================================
    # GÖRSELLEŞTİRME (OPSİYONEL)
    # =========================================================================
    if args.visualize:
        visualize_results(results, profession, args.output_dir)

    # =========================================================================
    # SONUÇLARI DOSYAYA KAYDET
    # =========================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, f'{profession}_{model_type}_test_results.json')

    # JSON'a kaydedilebilir formata çevir
    results_to_save = {
        'profession': profession,
        'model': model_type,
        'mae': float(results['mae']),
        'rmse': float(results['rmse']),
        'r2': float(results['r2']),
        'mape': float(results['mape']),
        'classification_counts': {k: int(v) for k, v in counts.items()},
        'num_samples': int(len(results['targets']))
    }

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nSonuçlar kaydedildi: {results_path}")

    return results


# =============================================================================
# TEK İLÇE TAHMİNİ
# =============================================================================
def predict_single(profession: str, model_type: str, features: dict,
                   checkpoint_dir: str = 'checkpoints'):
    """
    Tek bir ilçe için personel norm tahmini yap.

    Bu fonksiyon, yeni bir ilçe için özellikler verildiğinde
    ideal personel sayısını tahmin eder.

    ÖRNEK KULLANIM:
        features = {
            'nufus_18plus': 50000,
            'yuzolcum': 500,
            'toplam_hayvan_2024': 10000,
            ...
        }
        tahmin = predict_single('veteriner', 'mlp', features)

    Args:
        profession: Meslek türü
        model_type: Model mimarisi
        features: Özellik değerleri sözlüğü
        checkpoint_dir: Checkpoint dizini

    Returns:
        int: Tahmin edilen personel sayısı (minimum 1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli yükle
    checkpoint_path = os.path.join(checkpoint_dir, f'{profession}_{model_type}_best.pt')
    model, checkpoint = load_model(checkpoint_path, device)

    # Özellik isimlerini ve scaler'ı al
    feature_names = checkpoint['feature_names']
    scaler = checkpoint['scaler']

    # Özellik vektörü oluştur (doğru sırada)
    feature_vector = np.array([[features.get(name, 0) for name in feature_names]])

    # Özellikleri normalize et
    feature_scaled = scaler.transform(feature_vector)

    # Tahmin yap
    with torch.no_grad():
        input_tensor = torch.FloatTensor(feature_scaled).to(device)
        prediction = model(input_tensor).cpu().numpy()[0]

    # Minimum 1 personel döndür
    return max(1, round(prediction))


# =============================================================================
# ANA FONKSİYON
# =============================================================================
def main():
    """
    Ana test fonksiyonu.

    Komut satırı argümanlarına göre:
        - Tek meslek testi (--profession veteriner)
        - Tüm meslekler testi (--all)

    yapılır.
    """
    args = parse_args()

    if args.all:
        # =====================================================================
        # TÜM MESLEKLERİ VE MODELLERİ TEST ET
        # =====================================================================
        professions = ['veteriner', 'gida', 'ziraat']
        models = ['mlp', 'resnet', 'attention']
        all_results = {}

        for prof in professions:
            all_results[prof] = {}
            for model_type in models:
                # Checkpoint dosyası var mı kontrol et
                checkpoint_path = os.path.join(args.checkpoint_dir, f'{prof}_{model_type}_best.pt')
                if os.path.exists(checkpoint_path):
                    results = test_profession(prof, model_type, args)
                    if results:
                        all_results[prof][model_type] = results

        # Karşılaştırma tablosu yazdır
        if all_results:
            print("\n" + "="*80)
            print("TÜM MESLEKLERİN VE MODELLERİN KARŞILAŞTIRMASI")
            print("="*80)
            print(f"{'Meslek':<12} | {'Model':<10} | {'MAE':>8} | {'RMSE':>8} | {'R²':>8} | {'MAPE':>8}")
            print("-"*80)
            for prof in professions:
                for model_type in models:
                    if model_type in all_results.get(prof, {}):
                        res = all_results[prof][model_type]
                        print(f"{prof:<12} | {model_type:<10} | {res['mae']:>8.2f} | {res['rmse']:>8.2f} | "
                              f"{res['r2']:>8.4f} | {res['mape']:>7.1f}%")

            # En iyi modelleri göster
            print("\n" + "="*80)
            print("HER MESLEK İÇİN EN İYİ MODEL")
            print("="*80)
            for prof in professions:
                if all_results.get(prof):
                    best_model = max(all_results[prof].items(), key=lambda x: x[1]['r2'])
                    print(f"{prof.upper()}: {best_model[0].upper()} (R²={best_model[1]['r2']:.4f})")
    else:
        # =====================================================================
        # TEK MESLEK TEST ET
        # =====================================================================
        test_profession(args.profession, args.model, args)


# =============================================================================
# ANA GİRİŞ NOKTASI
# =============================================================================
if __name__ == "__main__":
    """
    Script doğrudan çalıştırıldığında bu blok çalışır.

    Kullanım örnekleri:
        # Tek meslek testi:
        python test.py --profession veteriner --model mlp

        # Görselleştirme ile:
        python test.py --profession gida --visualize

        # Tüm meslekler:
        python test.py --all
    """
    main()
