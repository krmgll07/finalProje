#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
EXTERNAL TEST SETİ DEĞERLENDİRME SCRİPTİ
================================================================================
Bu script, eğitim verisinden TAMAMEN AYRILMIŞ external test seti üzerinde
model performansını değerlendirir.

EXTERNAL TEST STRATEJİSİ:
    Yöntem 1: İl Bazlı Ayrım
        - Bazı iller (örn: 10 il) tamamen external test için ayrılır
        - Model bu illeri eğitim sırasında hiç görmez
        - Gerçek dünya senaryosunu simüle eder

    Yöntem 2: Rastgele İlçe Ayrımı (Hold-out)
        - İlçelerin %20'si baştan ayrılır
        - Eğitim sadece kalan %80 üzerinde yapılır
        - Standard external validation

ÇIKTI:
    - External test sonuçları
    - Internal vs External karşılaştırması
    - İl bazlı performans analizi

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
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# Proje modüllerini import et
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

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


class SimpleDataset(Dataset):
    """Basit PyTorch Dataset."""
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_and_prepare_data(profession, data_dir='.'):
    """
    Veriyi yükle ve hazırla.

    Returns:
        DataFrame, feature_columns, target_column
    """
    file_map = {
        'veteriner': 'veteriner_hekim_birlesitirilmis_veri.csv',
        'gida': 'gida_muhendisi_birlesitirilmis_veri.csv',
        'ziraat': 'ziraat_muhendisi_birlesitirilmis_veri.csv'
    }

    target_map = {
        'veteriner': 'veteriner_hekim',
        'gida': 'gida_muhendisi',
        'ziraat': 'ziraat_muhendisi'
    }

    # Ortak özellikler (tüm meslekler için)
    common_features = [
        'yuzolcum', 'merkez_ilce_flag',
        'nufus_18plus', 'nufus_yogunlugu_km2',
        'toplam_hayvan_2021', 'toplam_hayvan_2022', 'toplam_hayvan_2023', 'toplam_hayvan_2024',
        'toplam_hayvan_ortalama',
        'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022', 'tarim_alani_2023', 'tarim_alani_2024',
        'tarim_alani_ortalama', 'tarim_alani_trend_5y', 'tarim_alani_yuzolcum_orani',
        'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani'
    ]

    # Önce data/processed klasöründe ara, yoksa ana dizinde ara
    file_path = os.path.join(data_dir, 'data', 'processed', file_map[profession])
    if not os.path.exists(file_path):
        file_path = os.path.join(data_dir, file_map[profession])

    # Dosyayı yükle
    for encoding in ['utf-8-sig', 'utf-8', 'iso-8859-1', 'windows-1254']:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    return df, common_features, target_map[profession]


def prepare_features(df, feature_columns, target_column):
    """Özellikleri hazırla ve temizle."""
    # Mevcut sütunları filtrele
    available_features = [col for col in feature_columns if col in df.columns]

    X = df[available_features].copy()
    y = df[target_column].copy()

    # Sonsuz değerleri NaN yap
    X = X.replace([np.inf, -np.inf], np.nan)

    # NaN'ları doldur
    for col in X.columns:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col] = X[col].fillna(median_val)

    # Outlier clipping
    for col in X.columns:
        if '_orani' in col or 'islem' in col:
            q99 = X[col].quantile(0.99)
            q01 = X[col].quantile(0.01)
            X[col] = X[col].clip(lower=q01, upper=q99)

    # Hedef NaN'larını 0 yap
    y = y.fillna(0)

    return X.values, y.values, available_features


def train_model(X_train, y_train, X_val, y_val, input_dim, epochs=100, patience=15):
    """Model eğit ve döndür."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # DataLoader'lar
    train_dataset = SimpleDataset(X_train_scaled, y_train)
    val_dataset = SimpleDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = get_model('mlp', input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Eğitim
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_model_state)
    return model, scaler


def evaluate_model(model, X_test, y_test, scaler):
    """Model performansını değerlendir."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_test_scaled = scaler.transform(X_test)
    test_dataset = SimpleDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(targets),
        'predictions': predictions,
        'targets': targets
    }


def external_test_by_province(profession):
    """
    İL BAZLI EXTERNAL TEST

    10 ili tamamen external test için ayır.
    Model bu illeri eğitim sırasında hiç görmez.
    """
    print(f"\n{'='*60}")
    print(f"İL BAZLI EXTERNAL TEST - {profession.upper()}")
    print(f"{'='*60}")

    set_seed(42)

    # Veriyi yükle
    df, feature_columns, target_column = load_and_prepare_data(profession)

    # External test için ayrılacak iller (farklı bölgelerden)
    # Coğrafi çeşitlilik için seçildi
    external_provinces = [
        'Trabzon',      # Karadeniz
        'Şanlıurfa',    # Güneydoğu
        'Eskişehir',    # İç Anadolu
        'Muğla',        # Ege
        'Kayseri',      # İç Anadolu
        'Samsun',       # Karadeniz
        'Mardin',       # Güneydoğu
        'Çanakkale',    # Marmara
        'Erzurum',      # Doğu Anadolu
        'Adana'         # Akdeniz
    ]

    # İl adı sütununu bul
    il_column = None
    for col in ['il_adi', 'il', 'İl', 'IL']:
        if col in df.columns:
            il_column = col
            break

    if il_column is None:
        print("HATA: İl sütunu bulunamadı!")
        return None

    # External ve internal verileri ayır
    external_mask = df[il_column].isin(external_provinces)
    df_external = df[external_mask].copy()
    df_internal = df[~external_mask].copy()

    print(f"Toplam ilçe sayısı: {len(df)}")
    print(f"Internal (eğitim) ilçe sayısı: {len(df_internal)}")
    print(f"External (test) ilçe sayısı: {len(df_external)}")
    print(f"External iller: {external_provinces}")

    # Internal veriyi hazırla
    X_internal, y_internal, available_features = prepare_features(df_internal, feature_columns, target_column)
    X_external, y_external, _ = prepare_features(df_external, feature_columns, target_column)

    input_dim = X_internal.shape[1]

    # Internal veriyi train/val/test olarak böl
    X_trainval, X_internal_test, y_trainval, y_internal_test = train_test_split(
        X_internal, y_internal, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42  # 0.125 * 0.8 = 0.1
    )

    print(f"\nEğitim seti: {len(X_train)} ilçe")
    print(f"Validation seti: {len(X_val)} ilçe")
    print(f"Internal test seti: {len(X_internal_test)} ilçe")
    print(f"External test seti: {len(X_external)} ilçe")

    # Model eğit
    print("\nModel eğitiliyor...")
    model, scaler = train_model(X_train, y_train, X_val, y_val, input_dim)

    # Internal test değerlendirmesi
    print("\nInternal test değerlendiriliyor...")
    internal_results = evaluate_model(model, X_internal_test, y_internal_test, scaler)

    # External test değerlendirmesi
    print("External test değerlendiriliyor...")
    external_results = evaluate_model(model, X_external, y_external, scaler)

    # Sonuçları yazdır
    print("\n" + "-"*60)
    print("SONUÇLAR")
    print("-"*60)
    print(f"{'Metrik':<20} | {'Internal Test':>15} | {'External Test':>15}")
    print("-"*60)
    print(f"{'R² Skoru':<20} | {internal_results['r2']:>15.4f} | {external_results['r2']:>15.4f}")
    print(f"{'MAE':<20} | {internal_results['mae']:>15.2f} | {external_results['mae']:>15.2f}")
    print(f"{'RMSE':<20} | {internal_results['rmse']:>15.2f} | {external_results['rmse']:>15.2f}")
    print(f"{'Örnek Sayısı':<20} | {internal_results['n_samples']:>15} | {external_results['n_samples']:>15}")

    # İl bazlı external sonuçlar
    print("\n" + "-"*60)
    print("İL BAZLI EXTERNAL TEST SONUÇLARI")
    print("-"*60)

    for province in external_provinces:
        prov_mask = df[il_column] == province
        if prov_mask.sum() > 0:
            prov_df = df[prov_mask]
            X_prov, y_prov, _ = prepare_features(prov_df, feature_columns, target_column)

            if len(X_prov) > 0:
                prov_results = evaluate_model(model, X_prov, y_prov, scaler)
                print(f"{province:<15}: R²={prov_results['r2']:>7.4f}, MAE={prov_results['mae']:>6.2f}, n={prov_results['n_samples']}")

    return {
        'profession': profession,
        'internal': internal_results,
        'external': external_results,
        'external_provinces': external_provinces
    }


def external_test_holdout(profession):
    """
    HOLD-OUT EXTERNAL TEST

    İlçelerin %20'si baştan external test için ayrılır.
    Bu ilçeler eğitim sürecinde hiç kullanılmaz.
    """
    print(f"\n{'='*60}")
    print(f"HOLD-OUT EXTERNAL TEST - {profession.upper()}")
    print(f"{'='*60}")

    set_seed(42)

    # Veriyi yükle
    df, feature_columns, target_column = load_and_prepare_data(profession)
    X, y, available_features = prepare_features(df, feature_columns, target_column)

    input_dim = X.shape[1]

    # Önce %20'yi external için ayır
    X_internal, X_external, y_internal, y_external = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Internal veriyi train/val/test olarak böl
    X_trainval, X_internal_test, y_trainval, y_internal_test = train_test_split(
        X_internal, y_internal, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=42
    )

    print(f"Toplam ilçe sayısı: {len(X)}")
    print(f"Eğitim seti: {len(X_train)} ilçe")
    print(f"Validation seti: {len(X_val)} ilçe")
    print(f"Internal test seti: {len(X_internal_test)} ilçe")
    print(f"External test seti: {len(X_external)} ilçe")

    # Model eğit
    print("\nModel eğitiliyor...")
    model, scaler = train_model(X_train, y_train, X_val, y_val, input_dim)

    # Internal test değerlendirmesi
    print("\nInternal test değerlendiriliyor...")
    internal_results = evaluate_model(model, X_internal_test, y_internal_test, scaler)

    # External test değerlendirmesi
    print("External test değerlendiriliyor...")
    external_results = evaluate_model(model, X_external, y_external, scaler)

    # Sonuçları yazdır
    print("\n" + "-"*60)
    print("SONUÇLAR")
    print("-"*60)
    print(f"{'Metrik':<20} | {'Internal Test':>15} | {'External Test':>15}")
    print("-"*60)
    print(f"{'R² Skoru':<20} | {internal_results['r2']:>15.4f} | {external_results['r2']:>15.4f}")
    print(f"{'MAE':<20} | {internal_results['mae']:>15.2f} | {external_results['mae']:>15.2f}")
    print(f"{'RMSE':<20} | {internal_results['rmse']:>15.2f} | {external_results['rmse']:>15.2f}")
    print(f"{'Örnek Sayısı':<20} | {internal_results['n_samples']:>15} | {external_results['n_samples']:>15}")

    return {
        'profession': profession,
        'internal': internal_results,
        'external': external_results
    }


def main():
    """Ana fonksiyon - Tüm external testleri çalıştır."""

    print("="*80)
    print("EXTERNAL TEST SETİ DEĞERLENDİRME RAPORU")
    print("="*80)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    professions = ['veteriner', 'gida', 'ziraat']
    all_results = {
        'province_based': {},
        'holdout': {}
    }

    # İl bazlı external test
    print("\n" + "#"*80)
    print("# YÖNTEM 1: İL BAZLI EXTERNAL TEST")
    print("#"*80)

    for profession in professions:
        result = external_test_by_province(profession)
        if result:
            all_results['province_based'][profession] = {
                'internal_r2': result['internal']['r2'],
                'external_r2': result['external']['r2'],
                'internal_mae': result['internal']['mae'],
                'external_mae': result['external']['mae'],
                'external_provinces': result['external_provinces']
            }

    # Hold-out external test
    print("\n" + "#"*80)
    print("# YÖNTEM 2: HOLD-OUT EXTERNAL TEST")
    print("#"*80)

    for profession in professions:
        result = external_test_holdout(profession)
        if result:
            all_results['holdout'][profession] = {
                'internal_r2': result['internal']['r2'],
                'external_r2': result['external']['r2'],
                'internal_mae': result['internal']['mae'],
                'external_mae': result['external']['mae']
            }

    # Özet tablo
    print("\n" + "="*80)
    print("ÖZET KARŞILAŞTIRMA TABLOSU")
    print("="*80)

    print("\nİL BAZLI EXTERNAL TEST:")
    print("-"*70)
    print(f"{'Meslek':<15} | {'Internal R²':>12} | {'External R²':>12} | {'Fark':>10}")
    print("-"*70)
    for prof in professions:
        if prof in all_results['province_based']:
            r = all_results['province_based'][prof]
            diff = r['external_r2'] - r['internal_r2']
            print(f"{prof:<15} | {r['internal_r2']:>12.4f} | {r['external_r2']:>12.4f} | {diff:>+10.4f}")

    print("\nHOLD-OUT EXTERNAL TEST:")
    print("-"*70)
    print(f"{'Meslek':<15} | {'Internal R²':>12} | {'External R²':>12} | {'Fark':>10}")
    print("-"*70)
    for prof in professions:
        if prof in all_results['holdout']:
            r = all_results['holdout'][prof]
            diff = r['external_r2'] - r['internal_r2']
            print(f"{prof:<15} | {r['internal_r2']:>12.4f} | {r['external_r2']:>12.4f} | {diff:>+10.4f}")

    # Sonuçları JSON olarak kaydet (float32 -> float dönüşümü)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Sonuçları results klasörüne kaydet
    os.makedirs('results', exist_ok=True)

    serializable_results = convert_to_serializable(all_results)
    with open('results/external_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\nSonuçlar kaydedildi: results/external_test_results.json")

    # CSV olarak da kaydet
    rows = []
    for method in ['province_based', 'holdout']:
        for prof in professions:
            if prof in all_results[method]:
                r = all_results[method][prof]
                rows.append({
                    'method': method,
                    'profession': prof,
                    'internal_r2': r['internal_r2'],
                    'external_r2': r['external_r2'],
                    'internal_mae': r['internal_mae'],
                    'external_mae': r['external_mae']
                })

    pd.DataFrame(rows).to_csv('results/external_test_results.csv', index=False)
    print(f"Sonuçlar kaydedildi: results/external_test_results.csv")

    return all_results


if __name__ == "__main__":
    results = main()
