"""
Dataset module for Personnel Norm Prediction
=============================================
This module handles data loading, preprocessing, and PyTorch Dataset creation
for training deep learning models to predict optimal personnel allocation.

Includes data augmentation techniques:
- Gaussian noise injection
- Feature perturbation
- SMOTE-like synthetic sample generation
- Mixup augmentation
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os


class DataAugmenter:
    """
    Data augmentation for tabular data.

    Implements various augmentation techniques suitable for regression tasks:
    - Gaussian noise injection
    - SMOTE-like synthetic sample generation
    - Mixup augmentation
    - Feature jittering
    """

    def __init__(self, noise_level: float = 0.1, n_neighbors: int = 5):
        """
        Initialize the augmenter.

        Args:
            noise_level: Standard deviation of Gaussian noise (relative to feature std)
            n_neighbors: Number of neighbors for SMOTE-like generation
        """
        self.noise_level = noise_level
        self.n_neighbors = n_neighbors

    def add_gaussian_noise(self, X: np.ndarray, y: np.ndarray,
                           n_samples: int = None) -> tuple:
        """
        Add Gaussian noise to create synthetic samples.

        Args:
            X: Feature matrix
            y: Target values
            n_samples: Number of synthetic samples (default: same as original)

        Returns:
            Tuple of (augmented_X, augmented_y)
        """
        if n_samples is None:
            n_samples = len(X)

        # Randomly select samples to augment
        indices = np.random.choice(len(X), n_samples, replace=True)
        X_selected = X[indices].copy()
        y_selected = y[indices].copy()

        # Calculate feature-wise standard deviation
        feature_std = np.std(X, axis=0) + 1e-8

        # Add noise proportional to feature std
        noise = np.random.normal(0, self.noise_level, X_selected.shape) * feature_std
        X_augmented = X_selected + noise

        # Add small noise to targets as well (proportional to target std)
        target_std = np.std(y) + 1e-8
        y_noise = np.random.normal(0, self.noise_level * 0.5, y_selected.shape) * target_std
        y_augmented = np.maximum(y_selected + y_noise, 0.1)  # Keep positive

        return X_augmented, y_augmented

    def smote_like_augmentation(self, X: np.ndarray, y: np.ndarray,
                                 n_samples: int = None) -> tuple:
        """
        Generate synthetic samples using SMOTE-like interpolation.

        Args:
            X: Feature matrix
            y: Target values
            n_samples: Number of synthetic samples

        Returns:
            Tuple of (synthetic_X, synthetic_y)
        """
        if n_samples is None:
            n_samples = len(X) // 2

        # Fit nearest neighbors
        k = min(self.n_neighbors, len(X) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)

        synthetic_X = []
        synthetic_y = []

        for _ in range(n_samples):
            # Randomly select a sample
            idx = np.random.randint(len(X))
            sample = X[idx]
            target = y[idx]

            # Find neighbors
            distances, neighbors = nn.kneighbors([sample])
            neighbor_idx = neighbors[0, np.random.randint(1, k + 1)]

            # Interpolate
            alpha = np.random.uniform(0.1, 0.9)
            new_sample = sample + alpha * (X[neighbor_idx] - sample)
            new_target = target + alpha * (y[neighbor_idx] - target)

            synthetic_X.append(new_sample)
            synthetic_y.append(max(new_target, 0.1))

        return np.array(synthetic_X), np.array(synthetic_y)

    def mixup_augmentation(self, X: np.ndarray, y: np.ndarray,
                           n_samples: int = None, alpha: float = 0.4) -> tuple:
        """
        Mixup augmentation: linear interpolation between random pairs.

        Args:
            X: Feature matrix
            y: Target values
            n_samples: Number of synthetic samples
            alpha: Beta distribution parameter

        Returns:
            Tuple of (mixed_X, mixed_y)
        """
        if n_samples is None:
            n_samples = len(X) // 2

        mixed_X = []
        mixed_y = []

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

        return np.array(mixed_X), np.array(mixed_y)

    def feature_jittering(self, X: np.ndarray, y: np.ndarray,
                          jitter_ratio: float = 0.05) -> tuple:
        """
        Apply small random perturbations to features.

        Args:
            X: Feature matrix
            y: Target values
            jitter_ratio: Maximum relative perturbation

        Returns:
            Tuple of (jittered_X, y)
        """
        # Random multiplicative jitter
        jitter = 1 + np.random.uniform(-jitter_ratio, jitter_ratio, X.shape)
        X_jittered = X * jitter

        return X_jittered, y.copy()

    def augment(self, X: np.ndarray, y: np.ndarray,
                methods: list = None, augment_factor: float = 1.0) -> tuple:
        """
        Apply multiple augmentation methods.

        Args:
            X: Feature matrix
            y: Target values
            methods: List of augmentation methods to use
                     Options: 'noise', 'smote', 'mixup', 'jitter'
            augment_factor: How much to increase the dataset size

        Returns:
            Tuple of (augmented_X, augmented_y) including original data
        """
        if methods is None:
            methods = ['noise', 'smote', 'mixup']

        all_X = [X]
        all_y = [y]

        n_per_method = int(len(X) * augment_factor / len(methods))

        if 'noise' in methods and n_per_method > 0:
            aug_X, aug_y = self.add_gaussian_noise(X, y, n_per_method)
            all_X.append(aug_X)
            all_y.append(aug_y)

        if 'smote' in methods and n_per_method > 0:
            aug_X, aug_y = self.smote_like_augmentation(X, y, n_per_method)
            all_X.append(aug_X)
            all_y.append(aug_y)

        if 'mixup' in methods and n_per_method > 0:
            aug_X, aug_y = self.mixup_augmentation(X, y, n_per_method)
            all_X.append(aug_X)
            all_y.append(aug_y)

        if 'jitter' in methods:
            aug_X, aug_y = self.feature_jittering(X, y)
            all_X.append(aug_X)
            all_y.append(aug_y)

        return np.vstack(all_X), np.concatenate(all_y)


class PersonnelDataset(Dataset):
    """
    PyTorch Dataset for personnel norm prediction.

    This dataset loads preprocessed CSV files containing district-level
    features and personnel counts for different professions.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize the dataset.

        Args:
            features: Input features as numpy array (n_samples, n_features)
            targets: Target values as numpy array (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataPreprocessor:
    """
    Data preprocessor for loading and preparing data for training.

    Handles:
    - Loading CSV files with Turkish character encoding
    - Feature selection based on profession type
    - Missing value imputation
    - Feature scaling
    - Train/validation/test split
    """

    # Feature columns for each profession - EXPANDED to use all available features
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

    ZIRAAT_FEATURES = [
        # Temel demografik
        'yuzolcum', 'merkez_ilce_flag',
        'nufus_18plus', 'nufus_yogunlugu_km2',
        # Hayvan verileri
        'toplam_hayvan_2021', 'toplam_hayvan_2022', 'toplam_hayvan_2023', 'toplam_hayvan_2024',
        'toplam_hayvan_ortalama',
        'hayvan_nufus_orani', 'hayvan_yuzolcum_orani',
        # Tarim alani (tum yillar)
        'tarim_alani_2020', 'tarim_alani_2021', 'tarim_alani_2022', 'tarim_alani_2023', 'tarim_alani_2024',
        'tarim_alani_ortalama', 'tarim_alani_trend_5y', 'tarim_alani_yuzolcum_orani',
        # Tarimsal uretim
        'tarimsal_uretim_2021', 'tarimsal_uretim_2022', 'tarimsal_uretim_2023', 'tarimsal_uretim_2024',
        'tarimsal_uretim_ortalama',
        # Denetim
        'denetim_sayisi', 'denetim_nufus_orani', 'denetim_hayvan_orani',
        # Ziraat islem verileri
        'ziraat_islem_adet', 'ziraat_islem_sure_toplam',
        'ziraat_islem_nufus_orani', 'ziraat_islem_hayvan_orani', 'ziraat_islem_yuzolcum_orani',
        'ziraat_islem_tarim_alani_orani', 'ziraat_islem_uretim_orani',
        # Personel verimlilik
        'ziraat_muhendisi_yas_ortalam', 'ziraat_muhendisi_verimi'
    ]

    TARGET_COLUMNS = {
        'veteriner': 'veteriner_hekim',
        'gida': 'gida_muhendisi',
        'ziraat': 'ziraat_muhendisi'
    }

    def __init__(self, profession: str, data_dir: str = '.'):
        """
        Initialize the preprocessor.

        Args:
            profession: One of 'veteriner', 'gida', 'ziraat'
            data_dir: Directory containing CSV files
        """
        self.profession = profession.lower()
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_columns = self._get_feature_columns()
        self.target_column = self.TARGET_COLUMNS[self.profession]

    def _get_feature_columns(self):
        """Get feature columns based on profession type."""
        if self.profession == 'veteriner':
            return self.VETERINER_FEATURES
        elif self.profession == 'gida':
            return self.GIDA_FEATURES
        elif self.profession == 'ziraat':
            return self.ZIRAAT_FEATURES
        else:
            raise ValueError(f"Unknown profession: {self.profession}")

    def load_data(self):
        """
        Load and preprocess data from CSV file.

        Returns:
            DataFrame with loaded data
        """
        file_map = {
            'veteriner': 'veteriner_hekim_birlesitirilmis_veri.csv',
            'gida': 'gida_muhendisi_birlesitirilmis_veri.csv',
            'ziraat': 'ziraat_muhendisi_birlesitirilmis_veri.csv'
        }

        # Önce data/processed klasöründe ara, yoksa ana dizinde ara
        file_path = os.path.join(self.data_dir, 'data', 'processed', file_map[self.profession])
        if not os.path.exists(file_path):
            file_path = os.path.join(self.data_dir, file_map[self.profession])

        # Try different encodings
        for encoding in ['utf-8-sig', 'utf-8', 'iso-8859-1', 'windows-1254']:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        print(f"Loaded {len(df)} records from {file_path}")
        return df

    def prepare_features(self, df: pd.DataFrame):
        """
        Prepare feature matrix from DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features, targets, valid_indices)
        """
        # Select available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        print(f"Using {len(available_features)} features: {available_features}")

        # Extract features and target
        X = df[available_features].copy()
        y = df[self.target_column].copy()

        # Store column info
        self.available_features = available_features

        # Handle infinite values first
        X = X.replace([np.inf, -np.inf], np.nan)

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

        # Clip extreme outliers to 99th percentile for each column
        # This prevents extreme values from dominating the model
        for col in X.columns:
            if '_orani' in col or 'islem' in col:  # Ratio columns tend to have outliers
                q99 = X[col].quantile(0.99)
                q01 = X[col].quantile(0.01)
                X[col] = X[col].clip(lower=q01, upper=q99)

        # IMPORTANT: Fill NaN target values with 0
        # In personel_durum.csv, empty values mean "no personnel" (0), not missing data
        # This is verified by checking that yas_ortalam and verimi are also 0 for these rows
        y = y.fillna(0)

        # Filter valid samples (target >= 0 to include districts with 0 personnel)
        # We use >= 0 instead of > 0 to include all districts
        valid_mask = y.notna()  # All rows are valid after fillna
        X_valid = X[valid_mask].values
        y_valid = y[valid_mask].values

        # Final check for NaN in features
        nan_mask = ~np.isnan(X_valid).any(axis=1)
        X_valid = X_valid[nan_mask]
        y_valid = y_valid[nan_mask]

        print(f"Total samples (including 0 personnel): {len(y_valid)}")
        print(f"  - Districts with personnel > 0: {(y_valid > 0).sum()}")
        print(f"  - Districts with personnel = 0: {(y_valid == 0).sum()}")

        return X_valid, y_valid, valid_mask

    def create_dataloaders(self, batch_size: int = 32,
                           test_size: float = 0.2,
                           val_size: float = 0.1,
                           random_state: int = 42,
                           augment: bool = False,
                           augment_factor: float = 1.0,
                           augment_methods: list = None):
        """
        Create train, validation, and test dataloaders.

        Args:
            batch_size: Batch size for dataloaders
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            augment: Whether to apply data augmentation
            augment_factor: How much to increase training data (1.0 = double)
            augment_methods: List of augmentation methods ('noise', 'smote', 'mixup', 'jitter')

        Returns:
            Tuple of (train_loader, val_loader, test_loader, input_dim)
        """
        # Load and prepare data
        df = self.load_data()
        X, y, _ = self.prepare_features(df)

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size, random_state=random_state
        )

        # Apply data augmentation to training data only
        original_train_size = len(X_train)
        if augment:
            augmenter = DataAugmenter(noise_level=0.1, n_neighbors=5)
            X_train, y_train = augmenter.augment(
                X_train, y_train,
                methods=augment_methods,
                augment_factor=augment_factor
            )
            print(f"Data augmentation applied: {original_train_size} -> {len(X_train)} samples")

        # Scale features (fit on original + augmented training data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Create datasets
        train_dataset = PersonnelDataset(X_train_scaled, y_train)
        val_dataset = PersonnelDataset(X_val_scaled, y_val)
        test_dataset = PersonnelDataset(X_test_scaled, y_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = X_train_scaled.shape[1]

        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)} samples" +
              (f" (augmented from {original_train_size})" if augment else ""))
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Input dimension: {input_dim}")

        return train_loader, val_loader, test_loader, input_dim

    def get_scaler(self):
        """Return the fitted scaler for inference."""
        return self.scaler

    def get_feature_names(self):
        """Return the list of feature names used."""
        return self.available_features


def load_all_data(data_dir: str = '.'):
    """
    Load all three profession datasets.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Dictionary with dataloaders for each profession
    """
    professions = ['veteriner', 'gida', 'ziraat']
    data = {}

    for prof in professions:
        print(f"\n{'='*50}")
        print(f"Loading {prof.upper()} data...")
        print('='*50)

        preprocessor = DataPreprocessor(prof, data_dir)
        train_loader, val_loader, test_loader, input_dim = preprocessor.create_dataloaders()

        data[prof] = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'input_dim': input_dim,
            'preprocessor': preprocessor
        }

    return data


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Test single profession
    preprocessor = DataPreprocessor('veteriner', '.')
    train_loader, val_loader, test_loader, input_dim = preprocessor.create_dataloaders()

    # Get a batch
    for features, targets in train_loader:
        print(f"\nBatch shape: features={features.shape}, targets={targets.shape}")
        print(f"Feature sample: {features[0][:5]}...")
        print(f"Target sample: {targets[0]}")
        break
