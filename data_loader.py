# Data loading utilities for CT Slice, Dynamic Share and MNIST datasets
# Adapted from test_sampling.py

import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.sparse import coo_matrix
from pathlib import Path
import time
from torchvision import datasets, transforms
import urllib.request
import gzip
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def load_ctslice_for_experiments(data_dir='./data', max_samples=11000, imbalanced=True, 
                                 train_ratio=0.7, temporal_split=True, random_seed=42):
    """
    Load CT Slice dataset for pareto experiments with temporal split.
    
    Args:
        data_dir (str): Directory containing slice_localization_data.csv
        max_samples (int): Maximum number of samples (default: 11000 for 1000 base + 10000 updates)
        imbalanced (bool): Whether to create imbalanced patient distribution
        train_ratio (float): Training data ratio (default: 0.7 for 70% train, 30% test)
        temporal_split (bool): If True, use temporal split (earlier data for train, later for test)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        X_train (torch.Tensor): Training features (n x d)
        X_test (torch.Tensor): Test features (n_test x d)
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels
    """
    print(f'Loading CT Slice dataset from {data_dir}...')
    print(f'  Max samples: {max_samples}')
    print(f'  Imbalanced: {imbalanced}')
    
    # Read CSV file
    data_path = f'{data_dir}/slice_localization_data.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Expected location: {os.path.abspath(data_path)}\n"
            f"Please ensure the CT Slice dataset is in the correct location."
        )
    
    df = pd.read_csv(data_path)
    
    print(f'  Total data in CSV: {len(df)} rows')
    
    # Create imbalanced dataset if requested
    if imbalanced and len(df) > max_samples:
        patient_counts = df['patientId'].value_counts()
        print(f'\n  Creating imbalanced patient distribution...')
        print(f'  Total patients: {len(patient_counts)}')
        
        # 90% from top 10 patients, 10% from others
        top_n_patients = 10
        dominant_patients = patient_counts.index[:top_n_patients].tolist()
        dominant_data = df[df['patientId'].isin(dominant_patients)]
        other_patients_data = df[~df['patientId'].isin(dominant_patients)]
        
        n_from_dominant = int(max_samples * 0.9)
        n_from_others = max_samples - n_from_dominant
        
        # Sample from dominant patients
        if len(dominant_data) >= n_from_dominant:
            sampled_dominant = dominant_data.iloc[:n_from_dominant]
        else:
            # Need more patients
            top_n_patients = 15
            dominant_patients = patient_counts.index[:top_n_patients].tolist()
            dominant_data = df[df['patientId'].isin(dominant_patients)]
            other_patients_data = df[~df['patientId'].isin(dominant_patients)]
            sampled_dominant = dominant_data.iloc[:min(len(dominant_data), n_from_dominant)]
            n_from_dominant = len(sampled_dominant)
            n_from_others = max_samples - n_from_dominant
        
        # Sample from other patients
        sampled_others = other_patients_data.iloc[:min(len(other_patients_data), n_from_others)]
        
        # Combine
        df = pd.concat([sampled_dominant, sampled_others], ignore_index=True)
        
        print(f'  Sampled from top {top_n_patients} patients: {n_from_dominant} ({n_from_dominant/len(df)*100:.1f}%)')
        print(f'  Sampled from other patients: {len(df)-n_from_dominant} ({(len(df)-n_from_dominant)/len(df)*100:.1f}%)')
    elif len(df) > max_samples:
        # Balanced sampling
        df = df.iloc[:max_samples]
    
    # Extract features: columns from value0 to value383 (384 features)
    # Skip patientId (column 0) and reference (last column)
    feature_cols = [col for col in df.columns if col.startswith('value')]
    X = df[feature_cols].values.astype(np.float32)
    
    # Extract labels: reference column (continuous values)
    # Convert to binary classification using median as threshold
    y_continuous = df['reference'].values.astype(np.float32)
    y_median = np.median(y_continuous)
    
    # Binary labels: 1 if above median, 0 if below median
    y = (y_continuous > y_median).astype(np.float32)
    
    print(f'\n  Dataset shape: {X.shape}')
    print(f'  Features: {len(feature_cols)}')
    print(f'  Reference values: min={y_continuous.min():.2f}, max={y_continuous.max():.2f}, median={y_median:.2f}')
    print(f'  Binary labels (using median threshold): {np.sum(y == 1)} class 1, {np.sum(y == 0)} class 0')
    
    # # Normalize features using StandardScaler
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    # Convert labels to +1/-1 for logistic regression
    y_binary = 2 * y - 1  # 0 -> -1, 1 -> +1
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Split into train/test based on strategy
    n_total = len(X)
    # Use ceiling to ensure we get enough training samples
    n_train = int(np.ceil(train_ratio * n_total))
    
    if temporal_split:
        # Temporal split - first 70% for training, last 30% for testing
        # Don't shuffle data, keep original order (assuming data in CSV is time-ordered)
        print(f'\n  Using temporal split (train_ratio={train_ratio}):')
        print(f'    Training: first {n_train} samples ({train_ratio*100:.0f}%)')
        print(f'    Testing: last {n_total - n_train} samples ({(1-train_ratio)*100:.0f}%)')
    else:
        # Random split (original approach)
        print(f'\n  Using random split (train_ratio={train_ratio}):')
        shuffle_indices = np.random.permutation(len(X))
        X = X[shuffle_indices]
        y_binary = y_binary[shuffle_indices]
        y = y[shuffle_indices]
    
    X_train = X[:n_train]
    y_train = y_binary[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]  # Keep original 0/1 for test
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    # # L2 normalize rows
    # X_train_norms = X_train.norm(2, 1, keepdim=True)
    # X_train_norms[X_train_norms == 0] = 1.0  # Avoid division by zero
    # X_train = X_train / X_train_norms
    
    # X_test_norms = X_test.norm(2, 1, keepdim=True)
    # X_test_norms[X_test_norms == 0] = 1.0
    # X_test = X_test / X_test_norms
    
    print(f'\n  Train set: {X_train.shape[0]} samples')
    print(f'  Test set: {X_test.shape[0]} samples')
    print(f'  Feature dimension: {X_train.shape[1]}')
    
    return X_train, X_test, y_train, y_test


def load_dynamic_share_for_experiments(data_dir='./data', max_samples=20000,
                                       train_ratio=0.7, temporal_split=True, random_seed=42):
    """
    Load Dynamic Share dataset for pareto experiments.
    
    Args:
        data_dir (str): Directory containing dynamic_share/ folder
        max_samples (int): Maximum number of samples (default: 20000)
        train_ratio (float): Training data ratio (default: 0.7 for 70% train, 30% test)
        temporal_split (bool): If True, use temporal split (earlier data for train, later for test)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        X_train (torch.Tensor): Training features (n x d)
        X_test (torch.Tensor): Test features (n_test x d)
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels
    """
    print(f'Loading Dynamic Share dataset from {data_dir}...')
    print(f'  Max samples: {max_samples}')
    
    data_dir_path = Path(data_dir) / 'dynamic_share'
    
    # Check if directory exists
    if not data_dir_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir_path}\n"
            f"Expected location: {os.path.abspath(data_dir_path)}\n"
            f"Please ensure the Dynamic Share dataset is in the correct location."
        )
    
    # Get all txt files
    input_files = sorted(list(data_dir_path.glob('*.txt')))
    num_file = len(input_files)
    print(f'  Found {num_file} data files')
    
    # Pre-scan: calculate matrix size and number of non-zero elements
    size_A = 0
    nnz_A = 0
    
    for filepath in input_files:
        with open(filepath, 'r') as f:
            filetext = f.read()
            # Count colons (non-zero elements)
            nnz_A += filetext.count(':')
            # Count lines
            lines = filetext.strip().split('\n')
            size_A += len(lines)
    
    print(f'  Total matrix size: {size_A} rows, nnz: {nnz_A}')
    
    # Pre-allocate arrays
    b = np.zeros(size_A)
    col_idx = np.zeros(nnz_A, dtype=int)
    row_idx = np.zeros(nnz_A, dtype=int)
    val = np.zeros(nnz_A)
    
    row = 0
    b_ptr = 0
    val_ptr = 0
    
    # Read data
    print(f'  Loading data files...')
    tStart = time.time()
    
    for filepath in input_files:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) == 0:
                continue
            
            # First number is the label
            b[b_ptr] = float(parts[0])
            b_ptr += 1
            
            # Parse features: index:value
            for part in parts[1:]:
                if ':' in part:
                    idx, value = part.split(':')
                    col_idx[val_ptr] = int(idx)
                    val[val_ptr] = float(value)
                    row_idx[val_ptr] = row
                    val_ptr += 1
            
            row += 1
    
    tEnd = time.time() - tStart
    print(f'  Load time: {tEnd:.2f} seconds')
    
    # Build sparse matrix and convert to dense
    max_col = col_idx.max() + 1 if len(col_idx) > 0 else 482
    print(f'  Building matrix: {size_A} x {max_col}')
    
    A_sparse = coo_matrix((val, (row_idx, col_idx)), shape=(size_A, max_col))
    X = A_sparse.toarray()
    y_continuous = b
    
    # Convert to binary classification using median threshold (computed on FULL dataset)
    y_median_full = np.median(y_continuous)
    print(f'\n  Computing binary threshold on full dataset:')
    print(f'    Full dataset size: {len(y_continuous)} samples')
    print(f'    Label range: [{y_continuous.min():.2f}, {y_continuous.max():.2f}]')
    print(f'    Median threshold: {y_median_full:.2f}')
    
    # Limit sample count AFTER computing threshold
    if len(X) > max_samples:
        print(f'\n  Limiting dataset to first {max_samples} samples (original: {len(X)})')
        X = X[:max_samples]
        y_continuous = y_continuous[:max_samples]
    
    # Apply the threshold computed on full dataset
    y = (y_continuous > y_median_full).astype(np.float32)
    
    n, d = X.shape
    print(f'\n  Dataset shape: {X.shape}')
    print(f'  Non-zero elements: {np.count_nonzero(X)}')
    
    print(f'\n  Label statistics (after limiting):')
    print(f'    Range: [{y_continuous.min():.2f}, {y_continuous.max():.2f}]')
    print(f'    Mean: {y_continuous.mean():.2f}')
    print(f'    Std: {y_continuous.std():.2f}')
    print(f'    Applied threshold: {y_median_full:.2f} (from full dataset)')
    print(f'    Binary labels: {np.sum(y == 1)} class 1, {np.sum(y == 0)} class 0')
    print(f'    Class balance: {np.sum(y == 1)/len(y)*100:.1f}% class 1, {np.sum(y == 0)/len(y)*100:.1f}% class 0')
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert labels to +1/-1 for logistic regression
    y_binary = 2 * y - 1  # 0 -> -1, 1 -> +1
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Split into train/test
    n_total = len(X)
    n_train = int(np.ceil(train_ratio * n_total))
    
    if temporal_split:
        # Temporal split: first portion for train, last for test
        print(f'\n  Using temporal split (train_ratio={train_ratio}):')
        print(f'    Training: first {n_train} samples ({train_ratio*100:.0f}%)')
        print(f'    Testing: last {n_total - n_train} samples ({(1-train_ratio)*100:.0f}%)')
    else:
        # Random split
        print(f'\n  Using random split (train_ratio={train_ratio}):')
        shuffle_indices = np.random.permutation(len(X))
        X = X[shuffle_indices]
        y_binary = y_binary[shuffle_indices]
        y = y[shuffle_indices]
    
    X_train = X[:n_train]
    y_train = y_binary[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]  # Keep original 0/1 for test
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    print(f'\n  Train set: {X_train.shape[0]} samples')
    print(f'  Test set: {X_test.shape[0]} samples')
    print(f'  Feature dimension: {X_train.shape[1]}')
    
    return X_train, X_test, y_train, y_test


def load_mnist_for_experiments(data_dir='./data', max_samples=6000,
                                train_ratio=0.7, temporal_split=False, random_seed=42,
                                num_base=1000, incremental_imbalance_ratio=None, 
                                dominant_class=3, base_imbalance_ratio=None):
    """
    Load MNIST dataset (classes 3 and 8 only) for pareto experiments.
    Simplified version: uses MNIST train set for base+incremental, test set for testing.
    
    Args:
        data_dir (str): Directory to store MNIST data
        max_samples (int): Not used in simplified version
        train_ratio (float): Not used in simplified version
        temporal_split (bool): Not used for MNIST
        random_seed (int): Random seed for reproducibility
        num_base (int): Number of base training samples (default: 1000)
        incremental_imbalance_ratio (float): If specified, creates imbalanced incremental data.
                                             E.g., 0.9 means 90% of incremental data is dominant_class.
                                             If None, incremental data is balanced.
        dominant_class (int): Which class dominates in imbalanced data (3 or 8)
        base_imbalance_ratio (float): If specified, creates imbalanced base data.
                                      E.g., 0.9 means 90% of base data is dominant_class.
                                      If None, base data is balanced (random).
    
    Returns:
        X_train (torch.Tensor): Training features (n x 784)
        X_test (torch.Tensor): Test features (n_test x 784)
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels (0/1)
    """
    print(f'Loading MNIST dataset (classes 3 and 8 only) from {data_dir}...')
    print(f'  Base samples: {num_base}')
    if base_imbalance_ratio is not None:
        print(f'  Base imbalance ratio: {base_imbalance_ratio:.2f} (dominant class: {dominant_class})')
    if incremental_imbalance_ratio is not None:
        print(f'  Incremental imbalance ratio: {incremental_imbalance_ratio:.2f} (dominant class: {dominant_class})')
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Load MNIST training set
    trainset = datasets.MNIST(data_dir, train=True, transform=transforms.ToTensor(), download=True)
    # Load MNIST test set
    testset = datasets.MNIST(data_dir, train=False, transform=transforms.ToTensor(), download=True)
    
    # Process training set
    X_train_full = torch.zeros(len(trainset), 784)
    y_train_full = torch.zeros(len(trainset))
    for i in range(len(trainset)):
        x, y = trainset[i]
        X_train_full[i] = x.view(784) - 0.5  # Center by subtracting 0.5
        y_train_full[i] = y
    
    # Process test set
    X_test_full = torch.zeros(len(testset), 784)
    y_test_full = torch.zeros(len(testset))
    for i in range(len(testset)):
        x, y = testset[i]
        X_test_full[i] = x.view(784) - 0.5  # Center by subtracting 0.5
        y_test_full[i] = y
    
    # Filter training set: keep only classes 3 and 8
    train_mask = (y_train_full.eq(3) + y_train_full.eq(8)).gt(0)
    X_train_filtered = X_train_full[train_mask].numpy()
    y_train_filtered = y_train_full[train_mask].eq(3).float().numpy()  # 3->1, 8->0
    
    # Filter test set: keep only classes 3 and 8
    test_mask = (y_test_full.eq(3) + y_test_full.eq(8)).gt(0)
    X_test = X_test_full[test_mask]
    y_test = y_test_full[test_mask].eq(3).float()  # 3->1, 8->0
    
    print(f'\n  MNIST training set (classes 3 and 8): {len(X_train_filtered)} samples')
    print(f'    Class 3: {y_train_filtered.sum():.0f} samples')
    print(f'    Class 8: {(1-y_train_filtered).sum():.0f} samples')
    print(f'  MNIST test set (classes 3 and 8): {len(X_test)} samples')
    print(f'    Class 3: {y_test.sum().item():.0f} samples')
    print(f'    Class 8: {(1-y_test).sum().item():.0f} samples')
    
    # Determine incremental size from configs
    # The caller should provide num_incremental through max_samples - num_base
    n_incremental = max_samples - num_base
    
    # Shuffle training data indices
    all_train_indices = np.arange(len(X_train_filtered))
    np.random.shuffle(all_train_indices)
    
    # 1. Select base samples (balanced or imbalanced)
    if base_imbalance_ratio is not None and num_base > 0:
        # Create imbalanced base data
        # Separate all training data by class
        class_3_mask = (y_train_filtered == 1)
        class_8_mask = (y_train_filtered == 0)
        
        class_3_indices = np.where(class_3_mask)[0]
        class_8_indices = np.where(class_8_mask)[0]
        
        # Shuffle class indices
        np.random.shuffle(class_3_indices)
        np.random.shuffle(class_8_indices)
        
        # Calculate how many samples we need from each class for base
        if dominant_class == 3:
            num_base_class_3 = int(num_base * base_imbalance_ratio)
            num_base_class_8 = num_base - num_base_class_3
        else:  # dominant_class == 8
            num_base_class_8 = int(num_base * base_imbalance_ratio)
            num_base_class_3 = num_base - num_base_class_8
        
        # Check if we have enough samples
        if len(class_3_indices) < num_base_class_3:
            print(f'  Warning: Not enough class 3 samples for base. Need {num_base_class_3}, have {len(class_3_indices)}')
            num_base_class_3 = len(class_3_indices)
            num_base_class_8 = min(num_base - num_base_class_3, len(class_8_indices))
        
        if len(class_8_indices) < num_base_class_8:
            print(f'  Warning: Not enough class 8 samples for base. Need {num_base_class_8}, have {len(class_8_indices)}')
            num_base_class_8 = len(class_8_indices)
            num_base_class_3 = min(num_base - num_base_class_8, len(class_3_indices))
        
        # Select base samples
        base_class_3_indices = class_3_indices[:num_base_class_3]
        base_class_8_indices = class_8_indices[:num_base_class_8]
        
        # Combine and shuffle base indices
        base_indices = np.concatenate([base_class_3_indices, base_class_8_indices])
        np.random.shuffle(base_indices)
        
        print(f'\n  Base data (imbalanced):')
        print(f'    Class 3: {num_base_class_3} samples ({num_base_class_3/len(base_indices)*100:.1f}%)')
        print(f'    Class 8: {num_base_class_8} samples ({num_base_class_8/len(base_indices)*100:.1f}%)')
        
        # Remaining indices for incremental data
        remaining_indices = np.setdiff1d(all_train_indices, base_indices)
    else:
        # Random base selection (balanced)
        base_indices = all_train_indices[:num_base]
        remaining_indices = all_train_indices[num_base:]
    
    # 2. Select incremental samples (balanced or imbalanced)
    if incremental_imbalance_ratio is not None and n_incremental > 0:
        # Need to create imbalanced incremental data
        remaining_y = y_train_filtered[remaining_indices]
        remaining_class_3_mask = (remaining_y == 1)
        remaining_class_8_mask = (remaining_y == 0)
        
        remaining_class_3_indices = remaining_indices[remaining_class_3_mask]
        remaining_class_8_indices = remaining_indices[remaining_class_8_mask]
        
        print(f'\n  Available for incremental data:')
        print(f'    Class 3: {len(remaining_class_3_indices)} samples')
        print(f'    Class 8: {len(remaining_class_8_indices)} samples')
        
        # Calculate how many samples we need from each class
        if dominant_class == 3:
            num_inc_class_3 = int(n_incremental * incremental_imbalance_ratio)
            num_inc_class_8 = n_incremental - num_inc_class_3
        else:  # dominant_class == 8
            num_inc_class_8 = int(n_incremental * incremental_imbalance_ratio)
            num_inc_class_3 = n_incremental - num_inc_class_8
        
        # Check if we have enough samples
        if len(remaining_class_3_indices) < num_inc_class_3:
            print(f'  Warning: Not enough class 3 samples. Need {num_inc_class_3}, have {len(remaining_class_3_indices)}')
            num_inc_class_3 = len(remaining_class_3_indices)
            num_inc_class_8 = min(n_incremental - num_inc_class_3, len(remaining_class_8_indices))
        
        if len(remaining_class_8_indices) < num_inc_class_8:
            print(f'  Warning: Not enough class 8 samples. Need {num_inc_class_8}, have {len(remaining_class_8_indices)}')
            num_inc_class_8 = len(remaining_class_8_indices)
            num_inc_class_3 = min(n_incremental - num_inc_class_8, len(remaining_class_3_indices))
        
        # Select samples
        inc_class_3_indices = remaining_class_3_indices[:num_inc_class_3]
        inc_class_8_indices = remaining_class_8_indices[:num_inc_class_8]
        
        # Combine and shuffle
        inc_indices = np.concatenate([inc_class_3_indices, inc_class_8_indices])
        np.random.shuffle(inc_indices)
        
        print(f'\n  Incremental data (imbalanced):')
        print(f'    Class 3: {num_inc_class_3} samples ({num_inc_class_3/len(inc_indices)*100:.1f}%)')
        print(f'    Class 8: {num_inc_class_8} samples ({num_inc_class_8/len(inc_indices)*100:.1f}%)')
    else:
        # Random incremental data
        inc_indices = remaining_indices[:n_incremental]
    
    # Combine base and incremental indices
    train_indices = np.concatenate([base_indices, inc_indices])
    
    # Extract training data
    X_train = X_train_filtered[train_indices]
    y_train_01 = y_train_filtered[train_indices]  # 0/1 labels
    
    # Convert training labels to +1/-1 for logistic regression
    y_train = 2 * y_train_01 - 1  # 0 -> -1, 1 -> +1
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    
    # L2 normalize rows
    X_train_norms = X_train.norm(2, 1, keepdim=True)
    X_train_norms[X_train_norms == 0] = 1.0
    X_train = X_train / X_train_norms
    
    X_test_norms = X_test.norm(2, 1, keepdim=True)
    X_test_norms[X_test_norms == 0] = 1.0
    X_test = X_test / X_test_norms
    
    print(f'\n  Final dataset sizes:')
    print(f'    Train set (base + incremental): {X_train.shape[0]} samples')
    print(f'    Test set: {X_test.shape[0]} samples')
    print(f'    Feature dimension: {X_train.shape[1]}')
    
    # Statistics
    print(f'\n  Label statistics:')
    y_train_01_tensor = (y_train + 1) / 2  # Convert back to 0/1 for statistics
    
    # Base statistics
    y_base_01 = y_train_01_tensor[:num_base]
    base_type = 'imbalanced' if base_imbalance_ratio is not None else 'random'
    print(f'    Base ({num_base} samples, {base_type}):')
    print(f'      Class 3: {y_base_01.sum().item():.0f} ({y_base_01.mean()*100:.2f}%)')
    print(f'      Class 8: {(1-y_base_01).sum().item():.0f} ({(1-y_base_01).mean()*100:.2f}%)')
    
    # Incremental statistics
    if len(train_indices) > num_base:
        y_inc_01 = y_train_01_tensor[num_base:]
        label_type = 'imbalanced' if incremental_imbalance_ratio is not None else 'random'
        print(f'    Incremental ({len(y_inc_01):.0f} samples, {label_type}):')
        print(f'      Class 3: {y_inc_01.sum().item():.0f} ({y_inc_01.mean()*100:.2f}%)')
        print(f'      Class 8: {(1-y_inc_01).sum().item():.0f} ({(1-y_inc_01).mean()*100:.2f}%)')
    
    # Test statistics
    print(f'    Test ({len(y_test)} samples):')
    print(f'      Class 3: {y_test.sum().item():.0f} ({y_test.mean()*100:.2f}%)')
    print(f'      Class 8: {(1-y_test).sum().item():.0f} ({(1-y_test).mean()*100:.2f}%)')
    
    return X_train, X_test, y_train, y_test


def load_covertype_for_experiments(data_dir='./data', max_samples=20000,
                                    train_ratio=0.7, temporal_split=False, random_seed=42,
                                    binary_task='class2_vs_rest'):
    """
    Load Covertype (Forest Cover Type) dataset for pareto experiments.
    
    The Covertype dataset contains 581,012 samples with 54 features.
    Original labels are 1-7 representing different forest cover types.
    We convert this to binary classification (one class vs all others).
    
    Args:
        data_dir (str): Directory to store/read Covertype data
        max_samples (int): Maximum number of samples (default: 20000)
        train_ratio (float): Training data ratio (default: 0.7 for 70% train, 30% test)
        temporal_split (bool): If True, use temporal split; if False, use random split (default: False, 
                               as Covertype data has no temporal ordering)
        random_seed (int): Random seed for reproducibility
        binary_task (str): Binary classification task:
            - 'class1_vs_rest' or 'spruce_fir_vs_rest': Class 1 (Spruce/Fir) vs all other classes
            - 'class2_vs_rest' or 'lodgepole_vs_rest': Class 2 (Lodgepole Pine) vs all other classes
            Default: 'class2_vs_rest'
    
    Returns:
        X_train (torch.Tensor): Training features (n x 54), standardized
        X_test (torch.Tensor): Test features (n_test x 54), standardized
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels (0/1)
    """
    print(f'Loading Covertype dataset from {data_dir}...')
    print(f'  Max samples: {max_samples}')
    print(f'  Binary task: {binary_task}')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # File paths
    gz_path = os.path.join(data_dir, 'covtype.data.gz')
    csv_path = os.path.join(data_dir, 'covtype.data')
    
    # Download dataset if not present
    if not os.path.exists(gz_path) and not os.path.exists(csv_path):
        print(f'  Downloading Covertype dataset...')
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        try:
            urllib.request.urlretrieve(url, gz_path)
            print(f'  Download complete: {gz_path}')
        except Exception as e:
            raise Exception(f"Failed to download Covertype dataset: {e}")
    
    # Read data
    if os.path.exists(gz_path):
        print(f'  Reading compressed file: {gz_path}')
        with gzip.open(gz_path, 'rt') as f:
            df = pd.read_csv(f, header=None)
    elif os.path.exists(csv_path):
        print(f'  Reading uncompressed file: {csv_path}')
        df = pd.read_csv(csv_path, header=None)
    else:
        raise FileNotFoundError(f"Covertype data file not found in {data_dir}")
    
    print(f'  Total data loaded: {len(df)} rows, {df.shape[1]} columns')
    
    # Separate features and labels
    X = df.iloc[:, :-1].values.astype(np.float32)  # First 54 columns are features
    y_multiclass = df.iloc[:, -1].values.astype(np.int32)  # Last column is label (1-7)
    
    print(f'\n  Original dataset shape: {X.shape}')
    print(f'  Label distribution (classes 1-7):')
    for label in range(1, 8):
        count = np.sum(y_multiclass == label)
        print(f'    Class {label}: {count} samples ({count/len(y_multiclass)*100:.2f}%)')
    
    # Convert to binary classification based on task
    if binary_task == 'class1_vs_rest' or binary_task == 'spruce_fir_vs_rest':
        # Class 1 (Spruce/Fir) vs all other classes
        y = (y_multiclass == 1).astype(np.float32)
        task_name = "Spruce/Fir (class 1) vs all other classes"
    elif binary_task == 'class2_vs_rest' or binary_task == 'lodgepole_vs_rest':
        # Class 2 (Lodgepole Pine) vs all other classes
        y = (y_multiclass == 2).astype(np.float32)
        task_name = "Lodgepole Pine (class 2) vs all other classes"
    else:
        raise ValueError(f"Unknown binary_task: {binary_task}")
    
    print(f'\n  Binary classification task: {task_name}')
    print(f'  Binary labels: {np.sum(y == 1)} positive class ({np.sum(y == 1)/len(y)*100:.2f}%), '
          f'{np.sum(y == 0)} negative class ({np.sum(y == 0)/len(y)*100:.2f}%)')
    
    # Set random seed for reproducibility (before any sampling)
    np.random.seed(random_seed)
    
    # Limit to max_samples if needed - use random sampling
    if len(X) > max_samples:
        print(f'\n  Randomly sampling {max_samples} from {len(X)} samples...')
        sample_indices = np.random.choice(len(X), size=max_samples, replace=False)
        X = X[sample_indices]
        y = y[sample_indices]
        print(f'  After sampling: {np.sum(y == 1)} positive class ({np.sum(y == 1)/len(y)*100:.2f}%), '
              f'{np.sum(y == 0)} negative class ({np.sum(y == 0)/len(y)*100:.2f}%)')
    
    # Convert labels to +1/-1 for logistic regression
    y_binary = 2 * y - 1  # 0 -> -1, 1 -> +1
    
    # Split into train/test
    n_total = len(X)
    n_train = int(np.ceil(train_ratio * n_total))
    
    if temporal_split:
        # Temporal split: first portion for train, last for test
        print(f'\n  Using temporal split (train_ratio={train_ratio}):')
        print(f'    Training: first {n_train} samples ({train_ratio*100:.0f}%)')
        print(f'    Testing: last {n_total - n_train} samples ({(1-train_ratio)*100:.0f}%)')
    else:
        # Random split
        print(f'\n  Using random split (train_ratio={train_ratio}):')
        shuffle_indices = np.random.permutation(len(X))
        X = X[shuffle_indices]
        y_binary = y_binary[shuffle_indices]
        y = y[shuffle_indices]
    
    X_train = X[:n_train]
    y_train = y_binary[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]  # Keep original 0/1 for test
    
    # Standardize features (z-score normalization)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    print(f'\n  Train set: {X_train.shape[0]} samples')
    print(f'    Class 1: {((y_train + 1) / 2).sum().item():.0f}')
    print(f'    Class 0: {((y_train + 1) / 2).eq(0).sum().item():.0f}')
    print(f'  Test set: {X_test.shape[0]} samples')
    print(f'    Class 1: {y_test.sum().item():.0f}')
    print(f'    Class 0: {y_test.eq(0).sum().item():.0f}')
    print(f'  Feature dimension: {X_train.shape[1]}')
    
    return X_train, X_test, y_train, y_test


def load_a1a_for_experiments(data_dir='./data', max_samples=20000,
                              train_ratio=0.7, random_seed=42):
    """
    Load a1a dataset (from LIBSVM) for pareto experiments.
    
    The a1a dataset is a binary classification dataset from the UCI Adult dataset.
    Original files contain sparse features in LIBSVM format.
    
    Args:
        data_dir (str): Directory to store/read a1a data
        max_samples (int): Maximum number of samples (default: 20000)
        train_ratio (float): Training data ratio (default: 0.7 for 70% train, 30% test)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        X_train (torch.Tensor): Training features (n x 123), standardized
        X_test (torch.Tensor): Test features (n_test x 123), standardized
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels (0/1)
    """
    print(f'Loading a1a dataset from {data_dir}...')
    print(f'  Max samples: {max_samples}')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # File paths
    file_train = os.path.join(data_dir, 'a1a')
    file_test = os.path.join(data_dir, 'a1a.t')
    
    # Download dataset if not present
    base_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    
    for fname, full_path in [('a1a', file_train), ('a1a.t', file_test)]:
        if not os.path.exists(full_path):
            print(f'  Downloading {fname}...')
            try:
                urllib.request.urlretrieve(base_url + fname, full_path)
                print(f'    {fname} downloaded successfully')
            except Exception as e:
                raise Exception(f"Failed to download {fname}: {e}")
    
    # Load training and test data (sparse format)
    print(f'  Loading LIBSVM format data...')
    X_train_sparse, y_train_orig = load_svmlight_file(file_train, n_features=123)
    X_test_sparse, y_test_orig = load_svmlight_file(file_test, n_features=123)
    
    print(f'    Original train: {X_train_sparse.shape}')
    print(f'    Original test: {X_test_sparse.shape}')
    
    # Combine training and test data
    X_all = np.vstack([X_train_sparse.toarray(), X_test_sparse.toarray()])
    y_all = np.hstack([y_train_orig, y_test_orig])
    
    print(f'\n  Combined dataset: {X_all.shape}')
    
    # Convert labels: original is +1/-1, convert to 0/1
    y_all_01 = (y_all > 0).astype(np.float32)
    
    print(f'  Label distribution:')
    print(f'    Class 1: {np.sum(y_all_01 == 1)} samples ({np.sum(y_all_01 == 1)/len(y_all_01)*100:.2f}%)')
    print(f'    Class 0: {np.sum(y_all_01 == 0)} samples ({np.sum(y_all_01 == 0)/len(y_all_01)*100:.2f}%)')
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Limit to max_samples if needed - use random sampling
    if len(X_all) > max_samples:
        print(f'\n  Randomly sampling {max_samples} from {len(X_all)} samples...')
        sample_indices = np.random.choice(len(X_all), size=max_samples, replace=False)
        X_all = X_all[sample_indices]
        y_all_01 = y_all_01[sample_indices]
        print(f'  After sampling: {np.sum(y_all_01 == 1)} class 1 ({np.sum(y_all_01 == 1)/len(y_all_01)*100:.2f}%), '
              f'{np.sum(y_all_01 == 0)} class 0 ({np.sum(y_all_01 == 0)/len(y_all_01)*100:.2f}%)')
    
    # Split into train/test using stratified sampling
    print(f'\n  Splitting data (train_ratio={train_ratio}, stratified):')
    X_train, X_test, y_train_01, y_test = train_test_split(
        X_all, y_all_01,
        test_size=(1 - train_ratio),
        random_state=random_seed,
        stratify=y_all_01
    )
    
    # # Standardize features (z-score normalization)
    # print(f'\n  Applying feature standardization...')
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # Convert training labels to +1/-1 for logistic regression
    y_train = 2 * y_train_01 - 1  # 0 -> -1, 1 -> +1
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    
    print(f'\n  Train set: {X_train.shape[0]} samples')
    print(f'    Class 1: {((y_train + 1) / 2).sum().item():.0f}')
    print(f'    Class 0: {((y_train + 1) / 2).eq(0).sum().item():.0f}')
    print(f'  Test set: {X_test.shape[0]} samples')
    print(f'    Class 1: {y_test.sum().item():.0f}')
    print(f'    Class 0: {y_test.eq(0).sum().item():.0f}')
    print(f'  Feature dimension: {X_train.shape[1]}')
    
    return X_train, X_test, y_train, y_test


def load_w8a_for_experiments(data_dir='./data', max_samples=20000,
                              train_ratio=0.7, random_seed=42):
    """
    Load w8a dataset (from LIBSVM) for pareto experiments.
    
    The w8a dataset is a binary classification dataset.
    Original files contain sparse features in LIBSVM format.
    Uses the original train/test split from LIBSVM.
    
    Args:
        data_dir (str): Directory to store/read w8a data
        max_samples (int): Maximum number of training samples (default: 20000)
        train_ratio (float): Not used (keeps original train/test split)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        X_train (torch.Tensor): Training features (n x 300), standardized
        X_test (torch.Tensor): Test features (n_test x 300), standardized
        y_train (torch.Tensor): Training labels (+1/-1)
        y_test (torch.Tensor): Test labels (0/1)
    """
    print(f'Loading w8a dataset from {data_dir}...')
    print(f'  Max training samples: {max_samples}')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # File paths
    file_train = os.path.join(data_dir, 'w8a')
    file_test = os.path.join(data_dir, 'w8a.t')
    
    # Download dataset if not present
    base_url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    
    for fname, full_path in [('w8a', file_train), ('w8a.t', file_test)]:
        if not os.path.exists(full_path):
            print(f'  Downloading {fname}...')
            try:
                urllib.request.urlretrieve(base_url + fname, full_path)
                print(f'    {fname} downloaded successfully')
            except Exception as e:
                raise Exception(f"Failed to download {fname}: {e}")
    
    # Load training and test data (sparse format)
    print(f'  Loading LIBSVM format data...')
    n_features = 300  # w8a feature dimension
    X_train_sparse, y_train_orig = load_svmlight_file(file_train, n_features=n_features)
    X_test_sparse, y_test_orig = load_svmlight_file(file_test, n_features=n_features)
    
    print(f'    Original train: {X_train_sparse.shape}')
    print(f'    Original test: {X_test_sparse.shape}')
    
    # Convert to dense arrays
    X_train = X_train_sparse.toarray()
    X_test = X_test_sparse.toarray()
    
    # Convert labels: original is +1/-1, convert to 0/1 for easier handling
    y_train_01 = (y_train_orig > 0).astype(np.float32)
    y_test_01 = (y_test_orig > 0).astype(np.float32)
    
    print(f'\n  Original label distribution:')
    print(f'    Training: {np.sum(y_train_01 == 1)} class 1 ({np.sum(y_train_01 == 1)/len(y_train_01)*100:.2f}%), '
          f'{np.sum(y_train_01 == 0)} class 0 ({np.sum(y_train_01 == 0)/len(y_train_01)*100:.2f}%)')
    print(f'    Test: {np.sum(y_test_01 == 1)} class 1 ({np.sum(y_test_01 == 1)/len(y_test_01)*100:.2f}%), '
          f'{np.sum(y_test_01 == 0)} class 0 ({np.sum(y_test_01 == 0)/len(y_test_01)*100:.2f}%)')
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Limit training samples if needed - use random sampling
    if len(X_train) > max_samples:
        print(f'\n  Randomly sampling {max_samples} training samples from {len(X_train)}...')
        sample_indices = np.random.choice(len(X_train), size=max_samples, replace=False)
        X_train = X_train[sample_indices]
        y_train_01 = y_train_01[sample_indices]
        print(f'  After sampling: {np.sum(y_train_01 == 1)} class 1 ({np.sum(y_train_01 == 1)/len(y_train_01)*100:.2f}%), '
              f'{np.sum(y_train_01 == 0)} class 0 ({np.sum(y_train_01 == 0)/len(y_train_01)*100:.2f}%)')
    
    # # Standardize features (z-score normalization)
    # print(f'\n  Applying feature standardization...')
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # Convert training labels to +1/-1 for logistic regression
    y_train = 2 * y_train_01 - 1  # 0 -> -1, 1 -> +1
    
    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test_01).float()  # Keep test as 0/1
    
    print(f'\n  Train set: {X_train.shape[0]} samples')
    print(f'    Class 1: {((y_train + 1) / 2).sum().item():.0f}')
    print(f'    Class 0: {((y_train + 1) / 2).eq(0).sum().item():.0f}')
    print(f'  Test set: {X_test.shape[0]} samples')
    print(f'    Class 1: {y_test.sum().item():.0f}')
    print(f'    Class 0: {y_test.eq(0).sum().item():.0f}')
    print(f'  Feature dimension: {X_train.shape[1]}')
    
    return X_train, X_test, y_train, y_test


    
