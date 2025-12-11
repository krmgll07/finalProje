#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Based Personnel Surplus/Deficit Analysis for Turkish Districts
=================================================================
This script implements machine learning models to predict optimal personnel
allocation for veterinarians, food engineers, and agricultural engineers
across Turkish districts based on demographic, geographic, and operational data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================
# 1. Data Loading and Preprocessing
# =============================

def load_data():
    """Load all required datasets"""
    print("Loading datasets...")
    
    # Load CSV files with proper encoding for Turkish characters
    try:
        personnel = pd.read_csv("data/personel_durum.csv", encoding='iso-8859-1')
    except:
        personnel = pd.read_csv("data/personel_durum.csv", encoding='windows-1254')
    
    # Fix decimal separator issues (comma to dot)
    numeric_cols = ['gida_muhendisi', 'gida_muhendisi_yas_ortalam', 'gida_muhendisi_verimi',
                   'ziraat_muhendisi', 'ziraat_muhendisi_yas_ortalam', 'ziraat_muhendisi_yas_verimi',
                   'veteriner_hekim', 'veteriner_hekim_yas_ortalam', 'veteriner_hekim_yas_verimi']
    
    for col in numeric_cols:
        if col in personnel.columns:
            # Convert string numbers with commas to numeric
            personnel[col] = personnel[col].astype(str).str.replace(',', '.')
            personnel[col] = pd.to_numeric(personnel[col], errors='coerce')
    
    try:
        ilce = pd.read_csv("data/ilce_yuzolcum.csv", encoding='iso-8859-1')
    except:
        ilce = pd.read_csv("data/ilce_yuzolcum.csv", encoding='windows-1254')
    
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
    
    try:
        tarim_alani = pd.read_csv("data/tarim_alani.csv", encoding='iso-8859-1')
    except:
        tarim_alani = pd.read_csv("data/tarim_alani.csv", encoding='windows-1254')
    
    try:
        tarim_uretim = pd.read_csv("data/tarimsal_uretim.csv", encoding='iso-8859-1')
    except:
        tarim_uretim = pd.read_csv("data/tarimsal_uretim.csv", encoding='windows-1254')
    
    try:
        hayvan = pd.read_csv("data/canli_hayvan.csv", encoding='iso-8859-1')
    except:
        hayvan = pd.read_csv("data/canli_hayvan.csv", encoding='windows-1254')
    
    try:
        denetim = pd.read_csv("data/denetim_sayisi.csv", encoding='iso-8859-1')
    except:
        denetim = pd.read_csv("data/denetim_sayisi.csv", encoding='windows-1254')
    
    nufus = pd.read_excel("data/il_ilce_18yas_nufus.xlsx")
    
    print("All datasets loaded successfully!")
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
    """Preprocess population data to get 18+ population for 2024"""
    print("Preprocessing population data...")
    
    # Filter for 18+ population (Evet = Yes)
    nufus_18plus = nufus_df[nufus_df['gorunum_ad'].str.contains('Evet', na=False)].copy()
    
    # Get the latest year for each district
    nufus_18plus = nufus_18plus.sort_values(['il_kod', 'ilce_kod', 'yil'])
    nufus_last = nufus_18plus.groupby(['il_kod', 'ilce_kod']).last().reset_index()
    
    # Rename and select relevant columns
    nufus_last = nufus_last.rename(columns={'deger': 'nufus_18plus'})
    nufus_last = nufus_last[['il_kod', 'ilce_kod', 'nufus_18plus']]
    
    return nufus_last

def preprocess_tarim_alani_data(tarim_alani_df):
    """Preprocess agricultural area data"""
    print("Preprocessing agricultural area data...")
    
    # Calculate trend and 2024 values
    tarim_grouped = tarim_alani_df.groupby(['il_kod', 'ilce_kod'])[['y2020', 'y2021', 'y2022', 'y2023', 'y2024']].sum()
    tarim_grouped = tarim_grouped.reset_index()
    
    # Calculate trend (avoid division by zero)
    tarim_grouped['tarim_alani_2024'] = tarim_grouped['y2024']
    tarim_grouped['tarim_alani_ortalama'] = tarim_grouped[['y2020', 'y2021', 'y2022', 'y2023', 'y2024']].mean(axis=1)
    
    # Safe trend calculation
    y2020_safe = tarim_grouped['y2020'].replace({0: np.nan})
    tarim_grouped['tarim_alani_trend_5y'] = (tarim_grouped['y2024'] - tarim_grouped['y2020']) / y2020_safe
    tarim_grouped['tarim_alani_trend_5y'] = tarim_grouped['tarim_alani_trend_5y'].fillna(0)
    
    return tarim_grouped[['il_kod', 'ilce_kod', 'tarim_alani_2024', 'tarim_alani_trend_5y']]

def preprocess_tarim_uretim_data(tarim_uretim_df):
    """Preprocess agricultural production data"""
    print("Preprocessing agricultural production data...")
    
    # Sum production by district for 2024
    uretim_grouped = tarim_uretim_df.groupby(['il_kod', 'ilce_kod'])['y2024'].sum()
    uretim_grouped = uretim_grouped.reset_index()
    uretim_grouped = uretim_grouped.rename(columns={'y2024': 'tarimsal_uretim_2024_toplam'})
    
    return uretim_grouped

def preprocess_hayvan_data(hayvan_df):
    """Preprocess livestock data"""
    print("Preprocessing livestock data...")
    
    # Sum livestock by district for 2024
    hayvan_grouped = hayvan_df.groupby(['il_kod', 'ilce_kod'])['y2024'].sum()
    hayvan_grouped = hayvan_grouped.reset_index()
    hayvan_grouped = hayvan_grouped.rename(columns={'y2024': 'toplam_hayvan_2024'})
    
    return hayvan_grouped

def preprocess_norm_data(norm_df, profession_type):
    """Preprocess norm/workload data for different professions"""
    print(f"Preprocessing {profession_type} norm data...")
    
    # Group by district and sum operations
    norm_grouped = norm_df.groupby(['il_kod', 'ilce_kod'])[['islem_adeti', 'islem_suresi']].sum()
    norm_grouped = norm_grouped.reset_index()
    
    # Calculate total operation time
    norm_grouped[f'{profession_type}_islem_adet'] = norm_grouped['islem_adeti']
    norm_grouped[f'{profession_type}_islem_sure_toplam'] = norm_grouped['islem_adeti'] * norm_grouped['islem_suresi']
    
    return norm_grouped[['il_kod', 'ilce_kod', f'{profession_type}_islem_adet', f'{profession_type}_islem_sure_toplam']]

def preprocess_denetim_data(denetim_df):
    """Preprocess inspection data"""
    print("Preprocessing inspection data...")
    
    # Group by district and sum inspections
    denetim_grouped = denetim_df.groupby(['il_kod', 'ilce_kod'])['denetim_sayisi'].sum()
    denetim_grouped = denetim_grouped.reset_index()
    
    return denetim_grouped

def create_master_dataset(data_dict):
    """Create master dataset by merging all data sources"""
    print("Creating master dataset...")
    
    # Start with personnel data as base
    master_df = data_dict['personnel'].copy()
    
    # Debug: Check the data types and sample values
    print(f"Personnel data - il_kod type: {master_df['il_kod'].dtype}, sample values: {master_df['il_kod'].head()}")
    print(f"Personnel data - ilce_kod type: {master_df['ilce_kod'].dtype}, sample values: {master_df['ilce_kod'].head()}")
    
    # Ensure consistent data types for merging - keep as integers for better matching
    master_df['il_kod'] = pd.to_numeric(master_df['il_kod'], errors='coerce').astype('Int64')
    master_df['ilce_kod'] = pd.to_numeric(master_df['ilce_kod'], errors='coerce').astype('Int64')
    
    # Merge geographic data
    ilce_sel = data_dict['ilce'][['il_kod', 'ilce_kod', 'merkez_ilce_mi', 'yuzolcum']].copy()
    ilce_sel['il_kod'] = pd.to_numeric(ilce_sel['il_kod'], errors='coerce').astype('Int64')
    ilce_sel['ilce_kod'] = pd.to_numeric(ilce_sel['ilce_kod'], errors='coerce').astype('Int64')
    ilce_sel['merkez_ilce_flag'] = ilce_sel['merkez_ilce_mi'].map({'E': 1, 'H': 0} if ilce_sel['merkez_ilce_mi'].dtype == 'O' else lambda x: x)
    master_df = master_df.merge(ilce_sel, on=['il_kod', 'ilce_kod'], how='left')
    
    # Merge population data
    nufus_data = data_dict['nufus_processed'].copy()
    nufus_data['il_kod'] = pd.to_numeric(nufus_data['il_kod'], errors='coerce').astype('Int64')
    nufus_data['ilce_kod'] = pd.to_numeric(nufus_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(nufus_data, on=['il_kod', 'ilce_kod'], how='left')
    
    # Merge agricultural data
    tarim_alani_data = data_dict['tarim_alani_processed'].copy()
    tarim_alani_data['il_kod'] = pd.to_numeric(tarim_alani_data['il_kod'], errors='coerce').astype('Int64')
    tarim_alani_data['ilce_kod'] = pd.to_numeric(tarim_alani_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(tarim_alani_data, on=['il_kod', 'ilce_kod'], how='left')
    
    tarim_uretim_data = data_dict['tarim_uretim_processed'].copy()
    tarim_uretim_data['il_kod'] = pd.to_numeric(tarim_uretim_data['il_kod'], errors='coerce').astype('Int64')
    tarim_uretim_data['ilce_kod'] = pd.to_numeric(tarim_uretim_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(tarim_uretim_data, on=['il_kod', 'ilce_kod'], how='left')
    
    # Merge livestock data
    hayvan_data = data_dict['hayvan_processed'].copy()
    hayvan_data['il_kod'] = pd.to_numeric(hayvan_data['il_kod'], errors='coerce').astype('Int64')
    hayvan_data['ilce_kod'] = pd.to_numeric(hayvan_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(hayvan_data, on=['il_kod', 'ilce_kod'], how='left')
    
    # Merge inspection data
    denetim_data = data_dict['denetim_processed'].copy()
    denetim_data['il_kod'] = pd.to_numeric(denetim_data['il_kod'], errors='coerce').astype('Int64')
    denetim_data['ilce_kod'] = pd.to_numeric(denetim_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(denetim_data, on=['il_kod', 'ilce_kod'], how='left')
    
    # Merge norm data for all professions
    vet_norm_data = data_dict['vet_norm_processed'].copy()
    vet_norm_data['il_kod'] = pd.to_numeric(vet_norm_data['il_kod'], errors='coerce').astype('Int64')
    vet_norm_data['ilce_kod'] = pd.to_numeric(vet_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(vet_norm_data, on=['il_kod', 'ilce_kod'], how='left')
    
    gida_norm_data = data_dict['gida_norm_processed'].copy()
    gida_norm_data['il_kod'] = pd.to_numeric(gida_norm_data['il_kod'], errors='coerce').astype('Int64')
    gida_norm_data['ilce_kod'] = pd.to_numeric(gida_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(gida_norm_data, on=['il_kod', 'ilce_kod'], how='left')
    
    ziraat_norm_data = data_dict['ziraat_norm_processed'].copy()
    ziraat_norm_data['il_kod'] = pd.to_numeric(ziraat_norm_data['il_kod'], errors='coerce').astype('Int64')
    ziraat_norm_data['ilce_kod'] = pd.to_numeric(ziraat_norm_data['ilce_kod'], errors='coerce').astype('Int64')
    master_df = master_df.merge(ziraat_norm_data, on=['il_kod', 'ilce_kod'], how='left')
    
    # Debug: Check how many rows have data after merging
    print(f"After merging - districts with population data: {master_df['nufus_18plus'].notna().sum()}")
    print(f"After merging - districts with agricultural area data: {master_df['tarim_alani_2024'].notna().sum()}")
    print(f"After merging - districts with livestock data: {master_df['toplam_hayvan_2024'].notna().sum()}")
    
    return master_df

def create_derived_features(df):
    """Create derived features and ratios"""
    print("Creating derived features...")
    
    # Population density
    df['nufus_18plus_km2'] = df['nufus_18plus'] / df['yuzolcum'].replace({0: np.nan})
    
    # Livestock ratios
    df['hayvan_nufus_orani'] = df['toplam_hayvan_2024'] / df['nufus_18plus'].replace({0: np.nan})
    df['hayvan_yuzolcum_orani'] = df['toplam_hayvan_2024'] / df['yuzolcum'].replace({0: np.nan})
    
    # Operation ratios for each profession
    for profession in ['veteriner', 'gida', 'ziraat']:
        df[f'{profession}_islem_nufus_orani'] = df[f'{profession}_islem_adet'] / df['nufus_18plus'].replace({0: np.nan})
        df[f'{profession}_islem_hayvan_orani'] = df[f'{profession}_islem_adet'] / df['toplam_hayvan_2024'].replace({0: np.nan})
    
    # Inspection ratios
    df['denetim_nufus_orani'] = df['denetim_sayisi'] / df['nufus_18plus'].replace({0: np.nan})
    df['denetim_hayvan_orani'] = df['denetim_sayisi'] / df['toplam_hayvan_2024'].replace({0: np.nan})
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

# =============================
# 2. Model Training Functions
# =============================

def get_balanced_districts(df, profession, efficiency_col, target_col):
    """Select balanced districts for training based on efficiency percentiles"""
    print(f"Selecting balanced districts for {profession}...")
    
    # Convert efficiency column to numeric, handling string values
    if efficiency_col in df.columns:
        df[efficiency_col] = pd.to_numeric(df[efficiency_col], errors='coerce')
    
    # Convert target column to numeric
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Filter out districts with missing data - be more lenient with efficiency values
    valid_data = df[
        (df[efficiency_col].notna()) &
        (df[efficiency_col] >= 0) &  # Allow 0 efficiency values
        (df[target_col].notna()) &
        (df[target_col] > 0)
    ].copy()
    
    if len(valid_data) == 0:
        print(f"No valid data for {profession}")
        return pd.DataFrame()
    
    # Calculate percentiles - use wider range if data is too concentrated
    q1 = valid_data[efficiency_col].quantile(0.25)
    q3 = valid_data[efficiency_col].quantile(0.75)
    
    print(f"Efficiency range for balanced districts: {q1:.2f} - {q3:.2f}")
    print(f"Total valid districts: {len(valid_data)}")
    print(f"Efficiency statistics - Min: {valid_data[efficiency_col].min():.2f}, Max: {valid_data[efficiency_col].max():.2f}, Mean: {valid_data[efficiency_col].mean():.2f}")
    
    # If efficiency range is too narrow or all values are the same, use all valid data
    if q3 - q1 < 0.01 or q1 == q3:  # Very narrow range or all same values
        print("Efficiency range too narrow or all same values, using all valid data")
        return valid_data
    
    # Select balanced districts (25th-75th percentile)
    balanced_mask = valid_data[efficiency_col].between(q1, q3)
    balanced_districts = valid_data[balanced_mask].copy()
    
    print(f"Selected {len(balanced_districts)} balanced districts")
    
    # If too few, use broader range (10th-90th percentile)
    if len(balanced_districts) < 10:
        print("Too few balanced districts, using broader filter (10th-90th percentile)")
        q1 = valid_data[efficiency_col].quantile(0.10)
        q3 = valid_data[efficiency_col].quantile(0.90)
        balanced_mask = valid_data[efficiency_col].between(q1, q3)
        balanced_districts = valid_data[balanced_mask].copy()
        print(f"Selected {len(balanced_districts)} balanced districts with broader filter")
    
    # If still too few, use all valid data
    if len(balanced_districts) < 5:
        print("Still too few balanced districts, using all valid data")
        balanced_districts = valid_data.copy()
    
    return balanced_districts

def train_profession_model(df, profession, feature_cols):
    """Train ElasticNet model for a specific profession"""
    print(f"\nTraining model for {profession}...")
    
    # Handle different column naming conventions
    if profession == 'veteriner':
        target_col = 'veteriner_hekim'
        efficiency_col = 'veteriner_hekim_yas_verimi'
    elif profession == 'gida':
        target_col = 'gida_muhendisi'
        efficiency_col = 'gida_muhendisi_verimi'
    elif profession == 'ziraat':
        target_col = 'ziraat_muhendisi'
        efficiency_col = 'ziraat_muhendisi_yas_verimi'
    else:
        target_col = f'{profession}_muhendisi'
        efficiency_col = f'{profession}_muhendisi_verimi'
    
    # Check if columns exist
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found. Available columns: {list(df.columns)}")
        return None, None
    
    if efficiency_col not in df.columns:
        print(f"Warning: Efficiency column '{efficiency_col}' not found. Using target column for filtering.")
        efficiency_col = target_col
    
    # Get balanced training data
    train_df = get_balanced_districts(df, profession, efficiency_col, target_col)
    
    if len(train_df) == 0:
        print(f"Skipping {profession} due to insufficient data")
        return None, None
    
    # Prepare features and target
    X = train_df[feature_cols]
    y = train_df[target_col]
    
    # Debug: Check which features have missing values
    print("Missing values per feature:")
    for col in feature_cols:
        missing_count = X[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values")
    
    # Remove rows with missing values - be more lenient and drop features with too many missing values
    # First, drop features that have more than 50% missing values
    feature_missing_pct = X.isna().sum() / len(X)
    good_features = feature_missing_pct[feature_missing_pct < 0.5].index.tolist()
    
    if len(good_features) < len(feature_cols):
        print(f"Dropping features with >50% missing values: {set(feature_cols) - set(good_features)}")
        X = X[good_features]
    
    # Now remove rows with any remaining missing values
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"After removing missing values: {len(X)} samples")
    
    if len(X) < 5:  # Reduced minimum requirement
        print(f"Insufficient training data for {profession} (need at least 5 samples, have {len(X)})")
        return None, None
    
    print(f"Training data shape: {X.shape}")
    
    # Split data - use smaller test size if we have limited data
    test_size = 0.2 if len(X) >= 20 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Create and train model
    # Use stronger regularization for agricultural engineer due to extreme feature values
    if profession == 'ziraat':
        # For agricultural engineer, use stronger regularization and different alpha
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', ElasticNet(alpha=1.0, l1_ratio=0.7, random_state=42, max_iter=2000))
        ])
    else:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('reg', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
        ])
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance - MAE: {mae:.2f}, R²: {r2:.3f}")
    
    # Debug: Check for extreme values
    if mae > 1000 or r2 < -1:
        print(f"WARNING: Extreme values detected for {profession}")
        print(f"Target range: {y_test.min():.2f} to {y_test.max():.2f}")
        print(f"Prediction range: {y_pred.min():.2f} to {y_pred.max():.2f}")
        print(f"Feature statistics:")
        for col in X_train.columns:
            print(f"  {col}: {X_train[col].min():.2f} to {X_train[col].max():.2f}")
    
    return model, {
        'mae': mae,
        'r2': r2,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }, good_features  # Return the features that were actually used

def predict_all_districts(df, model, profession, feature_cols, good_features=None):
    """Make predictions for all districts"""
    print(f"Making predictions for {profession}...")
    
    # Get the correct target column name
    if profession == 'veteriner':
        target_col = 'veteriner_hekim'
    elif profession == 'gida':
        target_col = 'gida_muhendisi'
    elif profession == 'ziraat':
        target_col = 'ziraat_muhendisi'
    else:
        target_col = f'{profession}_muhendisi'
    
    # Use only the features that were actually used in training
    if good_features is not None:
        feature_cols_to_use = good_features
    else:
        feature_cols_to_use = feature_cols
    
    # Prepare features for all districts
    X_all = df[feature_cols_to_use].copy()
    
    # Make predictions
    predictions = model.predict(X_all)
    
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    
    # Create results dataframe
    results_df = df[['il_adi', 'ilce_adi', 'il_kod', 'ilce_kod', target_col]].copy()
    results_df['tahmini_norm'] = predictions
    results_df['tahmini_norm_yuvarlak'] = np.round(predictions)
    results_df['norm_farki'] = results_df[target_col] - results_df['tahmini_norm']
    
    return results_df

def classify_personnel_status(df, profession, target_col, min_abs_fark=1, min_goreli_fark=0.2):
    """Classify districts as surplus, deficit, or balanced"""
    
    def classify_row(row):
        norm = row['tahmini_norm']
        mevcut = row[target_col]
        fark = row['norm_farki']
        
        if pd.isna(norm) or pd.isna(mevcut) or norm < 0.5:
            return 'belirsiz'
        
        rel = abs(fark) / norm if norm > 0 else np.nan
        
        if (fark <= -min_abs_fark) and (rel >= min_goreli_fark):
            return 'norm_eksigi'
        elif (fark >= min_abs_fark) and (rel >= min_goreli_fark):
            return 'norm_fazlasi'
        else:
            return 'dengede'
    
    df[f'{profession}_durumu'] = df.apply(classify_row, axis=1)
    return df

# =============================
# 3. Main Execution Function
# =============================

def main():
    """Main execution function"""
    print("=" * 60)
    print("AI PERSONNEL SURPLUS/DEFICIT ANALYSIS SYSTEM")
    print("=" * 60)
    
    # Load data
    data_dict = load_data()
    
    # Preprocess individual datasets
    print("\nPreprocessing individual datasets...")
    data_dict['nufus_processed'] = preprocess_nufus_data(data_dict['nufus'])
    data_dict['tarim_alani_processed'] = preprocess_tarim_alani_data(data_dict['tarim_alani'])
    data_dict['tarim_uretim_processed'] = preprocess_tarim_uretim_data(data_dict['tarim_uretim'])
    data_dict['hayvan_processed'] = preprocess_hayvan_data(data_dict['hayvan'])
    data_dict['denetim_processed'] = preprocess_denetim_data(data_dict['denetim'])
    
    # Preprocess norm data for each profession
    data_dict['vet_norm_processed'] = preprocess_norm_data(data_dict['vet_norm'], 'veteriner')
    data_dict['gida_norm_processed'] = preprocess_norm_data(data_dict['gida_norm'], 'gida')
    data_dict['ziraat_norm_processed'] = preprocess_norm_data(data_dict['ziraat_norm'], 'ziraat')
    
    # Create master dataset
    master_df = create_master_dataset(data_dict)
    master_df = create_derived_features(master_df)
    
    print(f"\nMaster dataset created with {len(master_df)} districts")
    
    # Define feature columns for modeling
    feature_cols = [
        'merkez_ilce_flag', 'yuzolcum', 'nufus_18plus', 'nufus_18plus_km2',
        'tarim_alani_2024', 'tarim_alani_trend_5y', 'tarimsal_uretim_2024_toplam',
        'toplam_hayvan_2024', 'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        'veteriner_islem_adet', 'veteriner_islem_sure_toplam', 'veteriner_islem_nufus_orani',
        'veteriner_islem_hayvan_orani', 'gida_islem_adet', 'gida_islem_sure_toplam',
        'gida_islem_nufus_orani', 'ziraat_islem_adet', 'ziraat_islem_sure_toplam',
        'ziraat_islem_nufus_orani', 'denetim_sayisi', 'denetim_nufus_orani',
        'denetim_hayvan_orani'
    ]
    
    # Filter to only include available columns
    available_features = [col for col in feature_cols if col in master_df.columns]
    print(f"Available features: {len(available_features)}")
    
    # Train models for each profession
    professions = ['veteriner', 'gida', 'ziraat']
    models = {}
    results = {}
    model_features = {}  # Store the features used for each model
    
    for profession in professions:
        model, metrics, good_features = train_profession_model(master_df, profession, available_features)
        if model is not None:
            models[profession] = model
            results[profession] = metrics
            model_features[profession] = good_features  # Store the features used
            
            # Make predictions for all districts using only the features that were actually used
            pred_df = predict_all_districts(master_df, model, profession, available_features, good_features)
            
            # Merge predictions back to master dataframe
            master_df = master_df.merge(
                pred_df[['il_kod', 'ilce_kod', 'tahmini_norm', 'tahmini_norm_yuvarlak', 'norm_farki']],
                on=['il_kod', 'ilce_kod'], how='left'
            )
            
            # Rename the merged columns to avoid conflicts
            master_df[f'{profession}_tahmini_norm'] = master_df['tahmini_norm']
            master_df[f'{profession}_tahmini_norm_yuvarlak'] = master_df['tahmini_norm_yuvarlak']
            master_df[f'{profession}_norm_farki'] = master_df['norm_farki']
            
            # Clean up temporary columns so next profession doesn't conflict
            master_df = master_df.drop(columns=['tahmini_norm', 'tahmini_norm_yuvarlak', 'norm_farki'])
            
            # Classify personnel status with correct target column
            target_map = {
                'veteriner': 'veteriner_hekim',
                'gida': 'gida_muhendisi',
                'ziraat': 'ziraat_muhendisi'
            }
            target_col = target_map[profession]
            
            status_df = pred_df.copy()
            status_df = classify_personnel_status(status_df, profession, target_col)
            master_df = master_df.merge(
                status_df[['il_kod', 'ilce_kod', f'{profession}_durumu']],
                on=['il_kod', 'ilce_kod'], how='left'
            )
    
    # Generate summary reports
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORTS")
    print("=" * 60)
    
    for profession in professions:
        if profession in models:
            print(f"\n{profession.upper()} ANALYSIS:")
            print(f"Model Performance - MAE: {results[profession]['mae']:.2f}, R²: {results[profession]['r2']:.3f}")
            
            # Status distribution
            status_counts = master_df[f'{profession}_durumu'].value_counts()
            print(f"Status Distribution:")
            for status, count in status_counts.items():
                print(f"  {status}: {count} districts ({count/len(master_df)*100:.1f}%)")
            
            # Top 10 districts with largest deficits (most negative values)
            deficit_districts = master_df[master_df[f'{profession}_durumu'] == 'norm_eksigi'].nsmallest(10, f'{profession}_norm_farki')
            if len(deficit_districts) > 0:
                print(f"\nTop 10 Districts with Largest Deficits:")
                for _, row in deficit_districts.iterrows():
                    try:
                        print(f"  {row['il_adi']} {row['ilce_adi']}: {row[f'{profession}_norm_farki']:.1f} personnel deficit")
                    except UnicodeEncodeError:
                        # Handle Turkish characters that can't be encoded - use ASCII fallback
                        il_adi = row['il_adi'].encode('ascii', 'ignore').decode('ascii')
                        ilce_adi = row['ilce_adi'].encode('ascii', 'ignore').decode('ascii')
                        print(f"  {il_adi} {ilce_adi}: {row[f'{profession}_norm_farki']:.1f} personnel deficit")
            
            # Top 10 districts with largest surpluses
            surplus_districts = master_df[master_df[f'{profession}_durumu'] == 'norm_fazlasi'].nlargest(10, f'{profession}_norm_farki')
            if len(surplus_districts) > 0:
                print(f"\nTop 10 Districts with Largest Surpluses:")
                for _, row in surplus_districts.iterrows():
                    try:
                        print(f"  {row['il_adi']} {row['ilce_adi']}: +{row[f'{profession}_norm_farki']:.1f} personnel surplus")
                    except UnicodeEncodeError:
                        # Handle Turkish characters that can't be encoded - use ASCII fallback
                        il_adi = row['il_adi'].encode('ascii', 'ignore').decode('ascii')
                        ilce_adi = row['ilce_adi'].encode('ascii', 'ignore').decode('ascii')
                        print(f"  {il_adi} {ilce_adi}: +{row[f'{profession}_norm_farki']:.1f} personnel surplus")
    
    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    # Create summary dataframe
    summary_cols = ['il_adi', 'ilce_adi', 'nufus_18plus', 'yuzolcum', 'toplam_hayvan_2024']
    
    for profession in professions:
        if profession in models:
            if profession == 'veteriner':
                target_col = 'veteriner_hekim'
            else:
                target_col = f'{profession}_muhendisi'
            
            summary_cols.extend([
                target_col,
                f'{profession}_tahmini_norm_yuvarlak',
                f'{profession}_norm_farki',
                f'{profession}_durumu'
            ])
    
    summary_df = master_df[summary_cols].copy()
    
    # Export to CSV
    summary_df.to_csv('personnel_analysis_results.csv', index=False, encoding='utf-8')
    print("Results exported to 'personnel_analysis_results.csv'")
    
    # Export detailed results for each profession
    for profession in professions:
        if profession in models:
            if profession == 'veteriner':
                target_col = 'veteriner_hekim'
            else:
                target_col = f'{profession}_muhendisi'
            
            prof_cols = ['il_adi', 'ilce_adi', target_col,
                        f'{profession}_tahmini_norm', f'{profession}_tahmini_norm_yuvarlak',
                        f'{profession}_norm_farki', f'{profession}_durumu']
            prof_df = master_df[prof_cols].copy()
            prof_df = prof_df.sort_values(f'{profession}_norm_farki', ascending=False)
            prof_df.to_csv(f'{profession}_analysis_results.csv', index=False, encoding='utf-8')
            print(f"{profession.capitalize()} results exported to '{profession}_analysis_results.csv'")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return master_df, models, results

if __name__ == "__main__":
    # Run the analysis
    final_results, trained_models, model_metrics = main()