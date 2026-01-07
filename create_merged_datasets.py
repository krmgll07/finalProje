#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meslek Grubu Bazlı Birleştirilmiş Veri Setleri Oluşturma
========================================================
Bu script, Veteriner Hekim, Gıda Mühendisi ve Ziraat Mühendisi için
kullanılan tüm verileri birleştirip ayrı CSV dosyaları oluşturur.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Tüm veri setlerini yükle"""
    print("Veri setleri yükleniyor...")

    # Personel verisi
    try:
        personnel = pd.read_csv("data/personel_durum.csv", encoding='iso-8859-1')
    except:
        personnel = pd.read_csv("data/personel_durum.csv", encoding='windows-1254')

    # Ondalık ayırıcı düzeltme
    numeric_cols = ['gida_muhendisi', 'gida_muhendisi_yas_ortalam', 'gida_muhendisi_verimi',
                   'ziraat_muhendisi', 'ziraat_muhendisi_yas_ortalam', 'ziraat_muhendisi_yas_verimi',
                   'veteriner_hekim', 'veteriner_hekim_yas_ortalam', 'veteriner_hekim_yas_verimi']

    for col in numeric_cols:
        if col in personnel.columns:
            personnel[col] = personnel[col].astype(str).str.replace(',', '.')
            personnel[col] = pd.to_numeric(personnel[col], errors='coerce')

    # İlçe yüzölçüm verisi
    try:
        ilce = pd.read_csv("data/ilce_yuzolcum.csv", encoding='iso-8859-1')
    except:
        ilce = pd.read_csv("data/ilce_yuzolcum.csv", encoding='windows-1254')

    # Norm verileri
    try:
        vet_norm = pd.read_csv("data/norm_veteriner_hekim.csv", encoding='iso-8859-1')
    except:
        vet_norm = pd.read_csv("data/norm_veteriner_hekim.csv", encoding='windows-1254')

    try:
        gida_norm = pd.read_csv("data/norm_gida_mühendisi.csv", encoding='iso-8859-1')
    except:
        gida_norm = pd.read_csv("data/norm_gida_mühendisi.csv", encoding='windows-1254')

    try:
        ziraat_norm = pd.read_csv("data/norm_ziraat_mühendisi.csv", encoding='iso-8859-1')
    except:
        ziraat_norm = pd.read_csv("data/norm_ziraat_mühendisi.csv", encoding='windows-1254')

    # Tarım verileri
    try:
        tarim_alani = pd.read_csv("data/tarim_alani.csv", encoding='iso-8859-1')
    except:
        tarim_alani = pd.read_csv("data/tarim_alani.csv", encoding='windows-1254')

    try:
        tarim_uretim = pd.read_csv("data/tarimsal_uretim.csv", encoding='iso-8859-1')
    except:
        tarim_uretim = pd.read_csv("data/tarimsal_uretim.csv", encoding='windows-1254')

    # Hayvan verisi
    try:
        hayvan = pd.read_csv("data/canli_hayvan.csv", encoding='iso-8859-1')
    except:
        hayvan = pd.read_csv("data/canli_hayvan.csv", encoding='windows-1254')

    # Denetim verisi
    try:
        denetim = pd.read_csv("data/denetim_sayisi.csv", encoding='iso-8859-1')
    except:
        denetim = pd.read_csv("data/denetim_sayisi.csv", encoding='windows-1254')

    # Nüfus verisi
    nufus = pd.read_excel("data/il_ilce_18yas_nufus.xlsx")

    print("Tüm veri setleri başarıyla yüklendi!")
    return {
        'personnel': personnel,
        'ilce': ilce,
        'vet_norm': vet_norm,
        'gida_norm': gida_norm,
        'ziraat_norm': ziraat_norm,
        'tarim_alani': tarim_alani,
        'tarim_uretim': tarim_uretim,
        'hayvan': hayvan,
        'denetim': denetim,
        'nufus': nufus
    }


def preprocess_nufus_data(nufus_df):
    """18+ nüfus verisini işle"""
    nufus_18plus = nufus_df[nufus_df['gorunum_ad'].str.contains('Evet', na=False)].copy()
    nufus_18plus = nufus_18plus.sort_values(['il_kod', 'ilce_kod', 'yil'])
    nufus_last = nufus_18plus.groupby(['il_kod', 'ilce_kod']).last().reset_index()
    nufus_last = nufus_last.rename(columns={'deger': 'nufus_18plus'})
    nufus_last = nufus_last[['il_kod', 'ilce_kod', 'nufus_18plus']]
    return nufus_last


def preprocess_tarim_alani_data(tarim_alani_df):
    """Tarım alanı verisini işle"""
    # Mevcut yıl sütunlarını kontrol et
    year_cols = [col for col in ['y2020', 'y2021', 'y2022', 'y2023', 'y2024'] if col in tarim_alani_df.columns]
    tarim_grouped = tarim_alani_df.groupby(['il_kod', 'ilce_kod'])[year_cols].sum()
    tarim_grouped = tarim_grouped.reset_index()

    # Her yıl için sütun oluştur
    for col in year_cols:
        year = col.replace('y', '')
        tarim_grouped[f'tarim_alani_{year}'] = tarim_grouped[col]

    tarim_grouped['tarim_alani_ortalama'] = tarim_grouped[year_cols].mean(axis=1)

    # Trend hesapla (mevcut en eski ve en yeni yıla göre)
    if 'y2020' in year_cols and 'y2024' in year_cols:
        y2020_safe = tarim_grouped['y2020'].replace({0: np.nan})
        tarim_grouped['tarim_alani_trend_5y'] = (tarim_grouped['y2024'] - tarim_grouped['y2020']) / y2020_safe
    elif 'y2021' in year_cols and 'y2024' in year_cols:
        y2021_safe = tarim_grouped['y2021'].replace({0: np.nan})
        tarim_grouped['tarim_alani_trend_5y'] = (tarim_grouped['y2024'] - tarim_grouped['y2021']) / y2021_safe
    else:
        tarim_grouped['tarim_alani_trend_5y'] = 0
    tarim_grouped['tarim_alani_trend_5y'] = tarim_grouped['tarim_alani_trend_5y'].fillna(0)

    # Döndürülecek sütunları hazırla
    result_cols = ['il_kod', 'ilce_kod']
    for col in year_cols:
        year = col.replace('y', '')
        result_cols.append(f'tarim_alani_{year}')
    result_cols.extend(['tarim_alani_ortalama', 'tarim_alani_trend_5y'])

    return tarim_grouped[result_cols]


def preprocess_tarim_uretim_data(tarim_uretim_df):
    """Tarımsal üretim verisini işle"""
    # Mevcut yıl sütunlarını kontrol et
    year_cols = [col for col in ['y2020', 'y2021', 'y2022', 'y2023', 'y2024'] if col in tarim_uretim_df.columns]
    uretim_grouped = tarim_uretim_df.groupby(['il_kod', 'ilce_kod'])[year_cols].sum()
    uretim_grouped = uretim_grouped.reset_index()

    # Her yıl için sütun oluştur
    uretim_cols = []
    for col in year_cols:
        year = col.replace('y', '')
        new_col = f'tarimsal_uretim_{year}'
        uretim_grouped[new_col] = uretim_grouped[col]
        uretim_cols.append(new_col)

    uretim_grouped['tarimsal_uretim_ortalama'] = uretim_grouped[uretim_cols].mean(axis=1)

    # Sonuç sütunlarını hazırla
    result_cols = ['il_kod', 'ilce_kod'] + uretim_cols + ['tarimsal_uretim_ortalama']
    return uretim_grouped[result_cols]


def preprocess_hayvan_data(hayvan_df):
    """Canlı hayvan verisini işle"""
    # Mevcut yıl sütunlarını kontrol et
    year_cols = [col for col in ['y2020', 'y2021', 'y2022', 'y2023', 'y2024'] if col in hayvan_df.columns]
    hayvan_grouped = hayvan_df.groupby(['il_kod', 'ilce_kod'])[year_cols].sum()
    hayvan_grouped = hayvan_grouped.reset_index()

    # Her yıl için sütun oluştur
    hayvan_cols = []
    for col in year_cols:
        year = col.replace('y', '')
        new_col = f'toplam_hayvan_{year}'
        hayvan_grouped[new_col] = hayvan_grouped[col]
        hayvan_cols.append(new_col)

    hayvan_grouped['toplam_hayvan_ortalama'] = hayvan_grouped[hayvan_cols].mean(axis=1)

    # Sonuç sütunlarını hazırla
    result_cols = ['il_kod', 'ilce_kod'] + hayvan_cols + ['toplam_hayvan_ortalama']
    return hayvan_grouped[result_cols]


def preprocess_norm_data(norm_df, profession_type):
    """Norm/iş yükü verisini işle"""
    norm_grouped = norm_df.groupby(['il_kod', 'ilce_kod']).agg({
        'islem_adeti': 'sum',
        'islem_suresi': 'sum'
    }).reset_index()

    norm_grouped[f'{profession_type}_islem_adet'] = norm_grouped['islem_adeti']
    norm_grouped[f'{profession_type}_islem_sure_toplam'] = norm_grouped['islem_suresi']

    return norm_grouped[['il_kod', 'ilce_kod', f'{profession_type}_islem_adet', f'{profession_type}_islem_sure_toplam']]


def preprocess_denetim_data(denetim_df):
    """Denetim verisini işle"""
    denetim_grouped = denetim_df.groupby(['il_kod', 'ilce_kod'])['denetim_sayisi'].sum()
    denetim_grouped = denetim_grouped.reset_index()
    return denetim_grouped


def create_base_dataset(data_dict):
    """Temel veri setini oluştur"""
    print("Temel veri seti oluşturuluyor...")

    # Personel verisi ile başla
    master_df = data_dict['personnel'].copy()

    # Veri tiplerini düzelt
    master_df['il_kod'] = pd.to_numeric(master_df['il_kod'], errors='coerce').astype('Int64')
    master_df['ilce_kod'] = pd.to_numeric(master_df['ilce_kod'], errors='coerce').astype('Int64')

    # İlçe coğrafi verisi
    ilce_sel = data_dict['ilce'][['il_kod', 'ilce_kod', 'merkez_ilce_mi', 'yuzolcum']].copy()
    ilce_sel['il_kod'] = pd.to_numeric(ilce_sel['il_kod'], errors='coerce').astype('Int64')
    ilce_sel['ilce_kod'] = pd.to_numeric(ilce_sel['ilce_kod'], errors='coerce').astype('Int64')
    ilce_sel['merkez_ilce_flag'] = ilce_sel['merkez_ilce_mi'].map({'E': 1, 'H': 0})
    master_df = master_df.merge(ilce_sel, on=['il_kod', 'ilce_kod'], how='left')

    # Nüfus verisi
    nufus_data = data_dict['nufus_processed'].copy()
    nufus_data['il_kod'] = pd.to_numeric(nufus_data['il_kod'], errors='coerce').astype('Int64')
    nufus_data['ilce_kod'] = pd.to_numeric(nufus_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(nufus_data, on=['il_kod', 'ilce_kod'], how='left')

    # Tarım alanı verisi
    tarim_alani_data = data_dict['tarim_alani_processed'].copy()
    tarim_alani_data['il_kod'] = pd.to_numeric(tarim_alani_data['il_kod'], errors='coerce').astype('Int64')
    tarim_alani_data['ilce_kod'] = pd.to_numeric(tarim_alani_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(tarim_alani_data, on=['il_kod', 'ilce_kod'], how='left')

    # Tarımsal üretim verisi
    tarim_uretim_data = data_dict['tarim_uretim_processed'].copy()
    tarim_uretim_data['il_kod'] = pd.to_numeric(tarim_uretim_data['il_kod'], errors='coerce').astype('Int64')
    tarim_uretim_data['ilce_kod'] = pd.to_numeric(tarim_uretim_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(tarim_uretim_data, on=['il_kod', 'ilce_kod'], how='left')

    # Hayvan verisi
    hayvan_data = data_dict['hayvan_processed'].copy()
    hayvan_data['il_kod'] = pd.to_numeric(hayvan_data['il_kod'], errors='coerce').astype('Int64')
    hayvan_data['ilce_kod'] = pd.to_numeric(hayvan_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(hayvan_data, on=['il_kod', 'ilce_kod'], how='left')

    # Denetim verisi
    denetim_data = data_dict['denetim_processed'].copy()
    denetim_data['il_kod'] = pd.to_numeric(denetim_data['il_kod'], errors='coerce').astype('Int64')
    denetim_data['ilce_kod'] = pd.to_numeric(denetim_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(denetim_data, on=['il_kod', 'ilce_kod'], how='left')

    return master_df


def create_derived_features(df):
    """Türetilmiş özellikleri oluştur"""
    print("Türetilmiş özellikler hesaplanıyor...")

    # Nüfus yoğunluğu
    df['nufus_yogunlugu_km2'] = df['nufus_18plus'] / df['yuzolcum'].replace({0: np.nan})

    # En son hayvan yılı sütununu bul
    hayvan_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'toplam_hayvan_{year}'
        if col in df.columns:
            hayvan_col = col
            break

    # Hayvan oranları
    if hayvan_col:
        df['hayvan_nufus_orani'] = df[hayvan_col] / df['nufus_18plus'].replace({0: np.nan})
        df['hayvan_yuzolcum_orani'] = df[hayvan_col] / df['yuzolcum'].replace({0: np.nan})

    # En son tarım alanı yılı sütununu bul
    tarim_col = None
    for year in ['2024', '2023', '2022', '2021', '2020']:
        col = f'tarim_alani_{year}'
        if col in df.columns:
            tarim_col = col
            break

    # Tarım alanı oranları
    if tarim_col:
        df['tarim_alani_yuzolcum_orani'] = df[tarim_col] / df['yuzolcum'].replace({0: np.nan})

    # Denetim oranları
    if 'denetim_sayisi' in df.columns:
        df['denetim_nufus_orani'] = df['denetim_sayisi'] / df['nufus_18plus'].replace({0: np.nan})
        if hayvan_col:
            df['denetim_hayvan_orani'] = df['denetim_sayisi'] / df[hayvan_col].replace({0: np.nan})

    # Sonsuz değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def create_veteriner_dataset(master_df, data_dict):
    """Veteriner Hekim için birleştirilmiş veri seti"""
    print("\n" + "="*60)
    print("VETERİNER HEKİM VERİ SETİ OLUŞTURULUYOR")
    print("="*60)

    df = master_df.copy()

    # Veteriner norm verisi ekle
    vet_norm_data = data_dict['vet_norm_processed'].copy()
    vet_norm_data['il_kod'] = pd.to_numeric(vet_norm_data['il_kod'], errors='coerce').astype('Int64')
    vet_norm_data['ilce_kod'] = pd.to_numeric(vet_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    df = df.merge(vet_norm_data, on=['il_kod', 'ilce_kod'], how='left')

    # En son hayvan yılı sütununu bul
    hayvan_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'toplam_hayvan_{year}'
        if col in df.columns:
            hayvan_col = col
            break

    # Veteriner özel türetilmiş özellikler
    if 'veteriner_islem_adet' in df.columns:
        df['veteriner_islem_nufus_orani'] = df['veteriner_islem_adet'] / df['nufus_18plus'].replace({0: np.nan})
        if hayvan_col:
            df['veteriner_islem_hayvan_orani'] = df['veteriner_islem_adet'] / df[hayvan_col].replace({0: np.nan})
        df['veteriner_islem_yuzolcum_orani'] = df['veteriner_islem_adet'] / df['yuzolcum'].replace({0: np.nan})

    # Sonsuz değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)

    # Veteriner için ilgili sütunları seç ve sırala
    veteriner_cols = [
        # Kimlik bilgileri
        'il_kod', 'ilce_kod', 'il_adi', 'ilce_adi',
        # Coğrafi veriler
        'yuzolcum', 'merkez_ilce_mi', 'merkez_ilce_flag',
        # Nüfus verileri
        'nufus_18plus', 'nufus_yogunlugu_km2',
        # Hayvan verileri (Veteriner için önemli)
        'toplam_hayvan_2020', 'toplam_hayvan_2021', 'toplam_hayvan_2022',
        'toplam_hayvan_2023', 'toplam_hayvan_2024', 'toplam_hayvan_ortalama',
        'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        # Tarım verileri
        'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022',
        'tarim_alani_2023', 'tarim_alani_2024', 'tarim_alani_ortalama', 'tarim_alani_trend_5y',
        'tarim_alani_yuzolcum_orani',
        # Tarımsal üretim
        'tarimsal_uretim_2020', 'tarimsal_uretim_2021', 'tarimsal_uretim_2022',
        'tarimsal_uretim_2023', 'tarimsal_uretim_2024', 'tarimsal_uretim_ortalama',
        # Denetim verileri
        'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
        # Veteriner norm/iş yükü verileri
        'veteriner_islem_adet', 'veteriner_islem_sure_toplam',
        'veteriner_islem_nufus_orani', 'veteriner_islem_hayvan_orani', 'veteriner_islem_yuzolcum_orani',
        # Mevcut personel verileri
        'veteriner_hekim', 'veteriner_hekim_yas_ortalam', 'veteriner_hekim_yas_verimi'
    ]

    # Mevcut sütunları filtrele
    available_cols = [col for col in veteriner_cols if col in df.columns]
    df_veteriner = df[available_cols].copy()

    return df_veteriner


def create_gida_dataset(master_df, data_dict):
    """Gıda Mühendisi için birleştirilmiş veri seti"""
    print("\n" + "="*60)
    print("GIDA MÜHENDİSİ VERİ SETİ OLUŞTURULUYOR")
    print("="*60)

    df = master_df.copy()

    # Gıda norm verisi ekle
    gida_norm_data = data_dict['gida_norm_processed'].copy()
    gida_norm_data['il_kod'] = pd.to_numeric(gida_norm_data['il_kod'], errors='coerce').astype('Int64')
    gida_norm_data['ilce_kod'] = pd.to_numeric(gida_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    df = df.merge(gida_norm_data, on=['il_kod', 'ilce_kod'], how='left')

    # En son hayvan ve üretim yılı sütunlarını bul
    hayvan_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'toplam_hayvan_{year}'
        if col in df.columns:
            hayvan_col = col
            break

    uretim_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'tarimsal_uretim_{year}'
        if col in df.columns:
            uretim_col = col
            break

    # Gıda özel türetilmiş özellikler
    if 'gida_islem_adet' in df.columns:
        df['gida_islem_nufus_orani'] = df['gida_islem_adet'] / df['nufus_18plus'].replace({0: np.nan})
        if hayvan_col:
            df['gida_islem_hayvan_orani'] = df['gida_islem_adet'] / df[hayvan_col].replace({0: np.nan})
        df['gida_islem_yuzolcum_orani'] = df['gida_islem_adet'] / df['yuzolcum'].replace({0: np.nan})
        if uretim_col:
            df['gida_islem_uretim_orani'] = df['gida_islem_adet'] / df[uretim_col].replace({0: np.nan})

    # Sonsuz değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)

    # Gıda Mühendisi için ilgili sütunları seç ve sırala
    gida_cols = [
        # Kimlik bilgileri
        'il_kod', 'ilce_kod', 'il_adi', 'ilce_adi',
        # Coğrafi veriler
        'yuzolcum', 'merkez_ilce_mi', 'merkez_ilce_flag',
        # Nüfus verileri
        'nufus_18plus', 'nufus_yogunlugu_km2',
        # Hayvan verileri
        'toplam_hayvan_2020', 'toplam_hayvan_2021', 'toplam_hayvan_2022',
        'toplam_hayvan_2023', 'toplam_hayvan_2024', 'toplam_hayvan_ortalama',
        'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        # Tarım verileri
        'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022',
        'tarim_alani_2023', 'tarim_alani_2024', 'tarim_alani_ortalama', 'tarim_alani_trend_5y',
        'tarim_alani_yuzolcum_orani',
        # Tarımsal üretim (Gıda Mühendisi için önemli)
        'tarimsal_uretim_2020', 'tarimsal_uretim_2021', 'tarimsal_uretim_2022',
        'tarimsal_uretim_2023', 'tarimsal_uretim_2024', 'tarimsal_uretim_ortalama',
        # Denetim verileri (Gıda Mühendisi için önemli)
        'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
        # Gıda norm/iş yükü verileri
        'gida_islem_adet', 'gida_islem_sure_toplam',
        'gida_islem_nufus_orani', 'gida_islem_hayvan_orani', 'gida_islem_yuzolcum_orani', 'gida_islem_uretim_orani',
        # Mevcut personel verileri
        'gida_muhendisi', 'gida_muhendisi_yas_ortalam', 'gida_muhendisi_verimi'
    ]

    # Mevcut sütunları filtrele
    available_cols = [col for col in gida_cols if col in df.columns]
    df_gida = df[available_cols].copy()

    return df_gida


def create_ziraat_dataset(master_df, data_dict):
    """Ziraat Mühendisi için birleştirilmiş veri seti"""
    print("\n" + "="*60)
    print("ZİRAAT MÜHENDİSİ VERİ SETİ OLUŞTURULUYOR")
    print("="*60)

    df = master_df.copy()

    # Ziraat norm verisi ekle
    ziraat_norm_data = data_dict['ziraat_norm_processed'].copy()
    ziraat_norm_data['il_kod'] = pd.to_numeric(ziraat_norm_data['il_kod'], errors='coerce').astype('Int64')
    ziraat_norm_data['ilce_kod'] = pd.to_numeric(ziraat_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    df = df.merge(ziraat_norm_data, on=['il_kod', 'ilce_kod'], how='left')

    # En son hayvan, tarım alanı ve üretim yılı sütunlarını bul
    hayvan_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'toplam_hayvan_{year}'
        if col in df.columns:
            hayvan_col = col
            break

    tarim_col = None
    for year in ['2024', '2023', '2022', '2021', '2020']:
        col = f'tarim_alani_{year}'
        if col in df.columns:
            tarim_col = col
            break

    uretim_col = None
    for year in ['2024', '2023', '2022', '2021']:
        col = f'tarimsal_uretim_{year}'
        if col in df.columns:
            uretim_col = col
            break

    # Ziraat özel türetilmiş özellikler
    if 'ziraat_islem_adet' in df.columns:
        df['ziraat_islem_nufus_orani'] = df['ziraat_islem_adet'] / df['nufus_18plus'].replace({0: np.nan})
        if hayvan_col:
            df['ziraat_islem_hayvan_orani'] = df['ziraat_islem_adet'] / df[hayvan_col].replace({0: np.nan})
        df['ziraat_islem_yuzolcum_orani'] = df['ziraat_islem_adet'] / df['yuzolcum'].replace({0: np.nan})
        if tarim_col:
            df['ziraat_islem_tarim_alani_orani'] = df['ziraat_islem_adet'] / df[tarim_col].replace({0: np.nan})
        if uretim_col:
            df['ziraat_islem_uretim_orani'] = df['ziraat_islem_adet'] / df[uretim_col].replace({0: np.nan})

    # Sonsuz değerleri temizle
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ziraat Mühendisi için ilgili sütunları seç ve sırala
    ziraat_cols = [
        # Kimlik bilgileri
        'il_kod', 'ilce_kod', 'il_adi', 'ilce_adi',
        # Coğrafi veriler
        'yuzolcum', 'merkez_ilce_mi', 'merkez_ilce_flag',
        # Nüfus verileri
        'nufus_18plus', 'nufus_yogunlugu_km2',
        # Hayvan verileri
        'toplam_hayvan_2020', 'toplam_hayvan_2021', 'toplam_hayvan_2022',
        'toplam_hayvan_2023', 'toplam_hayvan_2024', 'toplam_hayvan_ortalama',
        'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        # Tarım verileri (Ziraat Mühendisi için çok önemli)
        'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022',
        'tarim_alani_2023', 'tarim_alani_2024', 'tarim_alani_ortalama', 'tarim_alani_trend_5y',
        'tarim_alani_yuzolcum_orani',
        # Tarımsal üretim (Ziraat Mühendisi için çok önemli)
        'tarimsal_uretim_2020', 'tarimsal_uretim_2021', 'tarimsal_uretim_2022',
        'tarimsal_uretim_2023', 'tarimsal_uretim_2024', 'tarimsal_uretim_ortalama',
        # Denetim verileri
        'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
        # Ziraat norm/iş yükü verileri
        'ziraat_islem_adet', 'ziraat_islem_sure_toplam',
        'ziraat_islem_nufus_orani', 'ziraat_islem_hayvan_orani', 'ziraat_islem_yuzolcum_orani',
        'ziraat_islem_tarim_alani_orani', 'ziraat_islem_uretim_orani',
        # Mevcut personel verileri
        'ziraat_muhendisi', 'ziraat_muhendisi_yas_ortalam', 'ziraat_muhendisi_yas_verimi'
    ]

    # Mevcut sütunları filtrele
    available_cols = [col for col in ziraat_cols if col in df.columns]
    df_ziraat = df[available_cols].copy()

    return df_ziraat


def main():
    """Ana çalıştırma fonksiyonu"""
    print("="*60)
    print("MESLEK GRUBU BAZLI BİRLEŞTİRİLMİŞ VERİ SETLERİ")
    print("="*60)

    # Verileri yükle
    data_dict = load_data()

    # Verileri ön işle
    print("\nVeriler ön işleniyor...")
    data_dict['nufus_processed'] = preprocess_nufus_data(data_dict['nufus'])
    data_dict['tarim_alani_processed'] = preprocess_tarim_alani_data(data_dict['tarim_alani'])
    data_dict['tarim_uretim_processed'] = preprocess_tarim_uretim_data(data_dict['tarim_uretim'])
    data_dict['hayvan_processed'] = preprocess_hayvan_data(data_dict['hayvan'])
    data_dict['denetim_processed'] = preprocess_denetim_data(data_dict['denetim'])
    data_dict['vet_norm_processed'] = preprocess_norm_data(data_dict['vet_norm'], 'veteriner')
    data_dict['gida_norm_processed'] = preprocess_norm_data(data_dict['gida_norm'], 'gida')
    data_dict['ziraat_norm_processed'] = preprocess_norm_data(data_dict['ziraat_norm'], 'ziraat')

    # Temel veri setini oluştur
    master_df = create_base_dataset(data_dict)
    master_df = create_derived_features(master_df)

    print(f"\nTemel veri seti oluşturuldu: {len(master_df)} ilçe")

    # Her meslek grubu için ayrı veri seti oluştur
    df_veteriner = create_veteriner_dataset(master_df, data_dict)
    df_gida = create_gida_dataset(master_df, data_dict)
    df_ziraat = create_ziraat_dataset(master_df, data_dict)

    # CSV dosyalarına kaydet
    print("\n" + "="*60)
    print("CSV DOSYALARI OLUŞTURULUYOR")
    print("="*60)

    df_veteriner.to_csv('veteriner_hekim_birlesitirilmis_veri.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] veteriner_hekim_birlesitirilmis_veri.csv ({len(df_veteriner)} satir, {len(df_veteriner.columns)} sutun)")

    df_gida.to_csv('gida_muhendisi_birlesitirilmis_veri.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] gida_muhendisi_birlesitirilmis_veri.csv ({len(df_gida)} satir, {len(df_gida.columns)} sutun)")

    df_ziraat.to_csv('ziraat_muhendisi_birlesitirilmis_veri.csv', index=False, encoding='utf-8-sig')
    print(f"[OK] ziraat_muhendisi_birlesitirilmis_veri.csv ({len(df_ziraat)} satir, {len(df_ziraat.columns)} sutun)")

    # Özet bilgileri yazdır
    print("\n" + "="*60)
    print("ÖZET BİLGİLER")
    print("="*60)

    print("\nVeteriner Hekim Veri Seti Sütunları:")
    print(f"  Toplam: {len(df_veteriner.columns)} sütun")

    print("\nGıda Mühendisi Veri Seti Sütunları:")
    print(f"  Toplam: {len(df_gida.columns)} sütun")

    print("\nZiraat Mühendisi Veri Seti Sütunları:")
    print(f"  Toplam: {len(df_ziraat.columns)} sütun")

    print("\n" + "="*60)
    print("İŞLEM TAMAMLANDI!")
    print("="*60)

    return df_veteriner, df_gida, df_ziraat


if __name__ == "__main__":
    df_vet, df_gida, df_ziraat = main()
