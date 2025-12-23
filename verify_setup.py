# Verification script to check setup and dependencies
# Run this before executing the main comparison

import sys
import os

def check_imports():
    """Check if all required packages are available"""
    print("Checking imports...")
    errors = []
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    - CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"    - CUDA not available (will use CPU)")
    except ImportError as e:
        errors.append(f"  ✗ PyTorch not found: {e}")
    
    try:
        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ NumPy not found: {e}")
    
    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ Matplotlib not found: {e}")
    
    try:
        import pandas as pd
        print(f"  ✓ Pandas {pd.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ Pandas not found: {e}")
    
    try:
        from sklearn.preprocessing import StandardScaler
        print(f"  ✓ Scikit-learn")
    except ImportError as e:
        errors.append(f"  ✗ Scikit-learn not found: {e}")
    
    # Check standard library
    try:
        import csv
        import argparse
        print(f"  ✓ Standard library (csv, argparse)")
    except ImportError as e:
        errors.append(f"  ✗ Standard library import failed: {e}")
    
    return errors


def check_files():
    """Check if all required files exist"""
    print("\nChecking files...")
    errors = []
    
    required_files = [
        'core_functions.py',
        'configs.py',
        'run_incremental.py',
        'run_streaming.py',
        'run_uniform.py',
        'compare_methods.py',
        'README.md',
        'VERSION.txt',
    ]
    
    for filename in required_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")
        else:
            errors.append(f"  ✗ {filename} not found")
    
    return errors


def check_parent_utils():
    """Check if parent directory utils are accessible"""
    print("\nChecking parent directory utilities...")
    errors = []
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    
    try:
        from utils import load_features
        print(f"  ✓ utils.load_features available")
    except ImportError as e:
        errors.append(f"  ✗ Cannot import utils.load_features: {e}")
    
    return errors


def check_directories():
    """Check/create output directories"""
    print("\nChecking output directories...")
    
    base_dir = os.path.dirname(__file__)
    dirs_to_check = [
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'results', 'csv'),
        os.path.join(base_dir, 'results', 'plots'),
    ]
    
    for directory in dirs_to_check:
        if os.path.exists(directory):
            print(f"  ✓ {os.path.relpath(directory, base_dir)} exists")
        else:
            os.makedirs(directory, exist_ok=True)
            print(f"  + {os.path.relpath(directory, base_dir)} created")
    
    return []


def check_configs():
    """Verify configurations are valid"""
    print("\nChecking configurations...")
    errors = []
    
    try:
        from configs import (
            DATASET_CONFIG, COMMON_ARGS, STREAMING_CONFIGS,
            UNIFORM_CONFIGS, EPSILON, OUTPUT_CONFIG
        )
        
        print(f"  ✓ Dataset: {DATASET_CONFIG['dataset']}")
        print(f"  ✓ Base training: {DATASET_CONFIG['num_training']}")
        print(f"  ✓ Update samples: {DATASET_CONFIG['num_removes']}")
        print(f"  ✓ Streaming configs: {len(STREAMING_CONFIGS)}")
        print(f"  ✓ Uniform configs: {len(UNIFORM_CONFIGS)}")
        print(f"  ✓ Total experiments: 1 + {len(STREAMING_CONFIGS)} + {len(UNIFORM_CONFIGS)} = {1 + len(STREAMING_CONFIGS) + len(UNIFORM_CONFIGS)}")
        
    except Exception as e:
        errors.append(f"  ✗ Error loading configs: {e}")
    
    return errors


def test_import_modules():
    """Test importing all experiment modules"""
    print("\nTesting module imports...")
    errors = []
    
    try:
        from core_functions import lr_loss, lr_optimize, spectral_norm
        print(f"  ✓ core_functions module")
    except Exception as e:
        errors.append(f"  ✗ core_functions import failed: {e}")
    
    try:
        from run_incremental import run_incremental_experiment
        print(f"  ✓ run_incremental module")
    except Exception as e:
        errors.append(f"  ✗ run_incremental import failed: {e}")
    
    try:
        from run_streaming import run_streaming_experiment
        print(f"  ✓ run_streaming module")
    except Exception as e:
        errors.append(f"  ✗ run_streaming import failed: {e}")
    
    try:
        from run_uniform import run_uniform_experiment
        print(f"  ✓ run_uniform module")
    except Exception as e:
        errors.append(f"  ✗ run_uniform import failed: {e}")
    
    return errors


def main():
    """Run all verification checks"""
    print("="*60)
    print("Pareto Experiments - Setup Verification")
    print("="*60)
    
    all_errors = []
    
    all_errors.extend(check_imports())
    all_errors.extend(check_files())
    all_errors.extend(check_parent_utils())
    all_errors.extend(check_directories())
    all_errors.extend(check_configs())
    all_errors.extend(test_import_modules())
    
    print("\n" + "="*60)
    if all_errors:
        print("VERIFICATION FAILED")
        print("="*60)
        print("\nErrors found:")
        for error in all_errors:
            print(error)
        return 1
    else:
        print("VERIFICATION PASSED")
        print("="*60)
        print("\nAll checks passed! You can now run:")
        print("  python compare_methods.py --data-dir <path_to_data>")
        return 0


if __name__ == '__main__':
    sys.exit(main())

