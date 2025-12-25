#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DATA AUGMENTATION KARŞILAŞTIRMA SCRIPTİ
================================================================================
Bu script, data augmentation kullanılan ve kullanılmayan durumların
sonuçlarını karşılaştırır.

KARŞILAŞTIRILAN DURUMLAR:
    1. Augmentation YOK (baseline)
    2. Gaussian Noise augmentation
    3. SMOTE-like augmentation
    4. Mixup augmentation
    5. Tüm yöntemler birlikte

ÇIKTI:
    - Karşılaştırmalı tablo (konsol + CSV)
    - Her meslek için ayrı sonuçlar

Yazar: Kerim GÜLLÜ
Tarih: Aralık 2024
Ders: AIE521 - Derin Öğrenme
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

# Proje modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import DataPreprocessor
from model import get_model, count_parameters


def set_seed(seed=42):
    """Tekrarlanabilirlik için rastgelelik tohumunu ayarla."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_evaluate(profession, model_type='mlp', augment=False,
                       augment_methods=None, augment_factor=1.0,
                       epochs=100, patience=15, verbose=False):
    """
    Modeli eğit ve değerlendir.

    Args:
        profession: Meslek türü (veteriner/gida/ziraat)
        model_type: Model mimarisi
        augment: Augmentation kullanılsın mı
        augment_methods: Kullanılacak augmentation yöntemleri
        augment_factor: Augmentation faktörü
        epochs: Maksimum epoch sayısı
        patience: Early stopping patience
        verbose: Detaylı çıktı

    Returns:
        dict: Eğitim ve test sonuçları
    """
    set_seed(42)

    # Cihaz seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Veriyi yükle
    preprocessor = DataPreprocessor(profession, '.')
    train_loader, val_loader, test_loader, input_dim = preprocessor.create_dataloaders(
        batch_size=32,
        augment=augment,
        augment_factor=augment_factor,
        augment_methods=augment_methods
    )

    train_size = len(train_loader.dataset)

    # Model oluştur
    if model_type == 'mlp':
        model_kwargs = {'hidden_dims': [128, 64, 32], 'dropout_rate': 0.3}
    else:
        model_kwargs = {'hidden_dim': 128, 'dropout_rate': 0.3}

    model = get_model(model_type, input_dim, **model_kwargs)
    model = model.to(device)

    # Eğitim bileşenleri
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Eğitim döngüsü
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping kontrolü
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    # En iyi modeli yükle
    model.load_state_dict(best_model_state)

    # Test değerlendirmesi
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Metrikler
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'train_size': train_size,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'epochs': epoch + 1
    }


def main():
    """Ana karşılaştırma fonksiyonu."""

    print("="*80)
    print("DATA AUGMENTATION KARŞILAŞTIRMA ANALİZİ")
    print("="*80)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test edilecek konfigürasyonlar
    augmentation_configs = [
        {'name': 'Augmentation YOK (Baseline)', 'augment': False, 'methods': None, 'factor': 0},
        {'name': 'Gaussian Noise', 'augment': True, 'methods': ['noise'], 'factor': 1.0},
        {'name': 'SMOTE-like', 'augment': True, 'methods': ['smote'], 'factor': 1.0},
        {'name': 'Mixup', 'augment': True, 'methods': ['mixup'], 'factor': 1.0},
        {'name': 'Tüm Yöntemler', 'augment': True, 'methods': ['noise', 'smote', 'mixup'], 'factor': 1.0},
    ]

    professions = ['veteriner', 'gida', 'ziraat']
    all_results = []

    for profession in professions:
        print(f"\n{'#'*80}")
        print(f"# {profession.upper()} MESLEĞİ")
        print(f"{'#'*80}")

        for config in augmentation_configs:
            print(f"\n  -> {config['name']}...")

            result = train_and_evaluate(
                profession=profession,
                model_type='mlp',
                augment=config['augment'],
                augment_methods=config['methods'],
                augment_factor=config['factor'],
                epochs=100,
                patience=15,
                verbose=False
            )

            result['profession'] = profession
            result['augmentation'] = config['name']
            all_results.append(result)

            print(f"     Train Size: {result['train_size']}, R²: {result['r2']:.4f}, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")

    # Sonuçları DataFrame'e çevir
    results_df = pd.DataFrame(all_results)

    # Karşılaştırma tablosu oluştur
    print("\n")
    print("="*100)
    print("KARŞILAŞTIRMA TABLOSU")
    print("="*100)

    for profession in professions:
        prof_results = results_df[results_df['profession'] == profession]

        print(f"\n{profession.upper()}")
        print("-"*80)
        print(f"{'Augmentation Yöntemi':<30} | {'Train Size':>10} | {'R²':>8} | {'MAE':>8} | {'RMSE':>8}")
        print("-"*80)

        baseline_r2 = prof_results[prof_results['augmentation'] == 'Augmentation YOK (Baseline)']['r2'].values[0]

        for _, row in prof_results.iterrows():
            r2_diff = row['r2'] - baseline_r2
            diff_str = f"({r2_diff:+.4f})" if row['augmentation'] != 'Augmentation YOK (Baseline)' else ""
            print(f"{row['augmentation']:<30} | {row['train_size']:>10} | {row['r2']:>8.4f} | {row['mae']:>8.2f} | {row['rmse']:>8.2f} {diff_str}")

    # En iyi sonuçları göster
    print("\n")
    print("="*80)
    print("EN İYİ SONUÇLAR (Her Meslek İçin)")
    print("="*80)

    for profession in professions:
        prof_results = results_df[results_df['profession'] == profession]
        best_row = prof_results.loc[prof_results['r2'].idxmax()]
        print(f"{profession.upper()}: {best_row['augmentation']} -> R²={best_row['r2']:.4f}")

    # Sonuçları results klasörüne kaydet
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/augmentation_comparison_results.csv', index=False)
    print(f"\nSonuçlar kaydedildi: results/augmentation_comparison_results.csv")

    # JSON olarak da kaydet
    summary = {
        'timestamp': datetime.now().isoformat(),
        'professions': {}
    }

    for profession in professions:
        prof_results = results_df[results_df['profession'] == profession]
        summary['professions'][profession] = {
            'baseline': prof_results[prof_results['augmentation'] == 'Augmentation YOK (Baseline)'].iloc[0].to_dict(),
            'best': prof_results.loc[prof_results['r2'].idxmax()].to_dict(),
            'all_results': prof_results.to_dict('records')
        }

    with open('results/augmentation_comparison_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"Özet kaydedildi: results/augmentation_comparison_summary.json")

    return results_df


if __name__ == "__main__":
    results = main()
