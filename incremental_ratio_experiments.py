# Incremental Data Ratio Experiments
# Tests incremental, uniform, and streaming methods with different ratios of incremental data
# Datasets: MNIST, CT Slice, Dynamic Share, W8A
# Data ratios: 10%, 30%, 50%, 70%, 90%, 100%

import sys
import os
import csv
import argparse
import pandas as pd
import torch
import time

# Import experiment runners and utilities
from run_incremental import run_incremental_experiment
from run_streaming import run_streaming_experiment
from run_uniform import run_uniform_experiment
from data_loader import (
    load_ctslice_for_experiments, 
    load_dynamic_share_for_experiments, 
    load_mnist_for_experiments,
    load_w8a_for_experiments
)
from configs import (
    DATASET_CONFIG, DATASET_CONFIG_DYNAMIC, DATASET_CONFIG_MNIST, DATASET_CONFIG_W8A,
    COMMON_ARGS,
    INCREMENTAL_DATA_RATIOS,
    INCREMENTAL_RATIO_NUM_STEPS,
    INCREMENTAL_RATIO_CONFIG_CT_SLICE,
    INCREMENTAL_RATIO_CONFIG_DYNAMIC,
    INCREMENTAL_RATIO_CONFIG_MNIST,
    INCREMENTAL_RATIO_CONFIG_W8A,
    OUTPUT_CONFIG_INCREMENTAL_RATIO
)
from core_functions import (
    lr_optimize, lr_optimize_newton, lr_optimize_lbfgs_torchmin,
    lr_optimize_bfgs, lr_optimize_trust_exact,
    lr_optimize_subsampled_newton_lev, lr_optimize_subsampled_newton_uniform,
    device
)


def get_dataset_configs(dataset_name):
    """Get dataset-specific configurations"""
    if dataset_name == 'ct_slice':
        return {
            'config': DATASET_CONFIG,
            'ratio_config': INCREMENTAL_RATIO_CONFIG_CT_SLICE,
            'loader': load_ctslice_for_experiments,
            'name': 'CT Slice',
            'imbalanced': True,
        }
    elif dataset_name == 'dynamic_share':
        return {
            'config': DATASET_CONFIG_DYNAMIC,
            'ratio_config': INCREMENTAL_RATIO_CONFIG_DYNAMIC,
            'loader': load_dynamic_share_for_experiments,
            'name': 'Dynamic Share',
            'imbalanced': False,
        }
    elif dataset_name == 'mnist':
        return {
            'config': DATASET_CONFIG_MNIST,
            'ratio_config': INCREMENTAL_RATIO_CONFIG_MNIST,
            'loader': load_mnist_for_experiments,
            'name': 'MNIST (Classes 3 and 8)',
            'imbalanced': False,
        }
    elif dataset_name == 'w8a':
        return {
            'config': DATASET_CONFIG_W8A,
            'ratio_config': INCREMENTAL_RATIO_CONFIG_W8A,
            'loader': load_w8a_for_experiments,
            'name': 'W8A Dataset',
            'imbalanced': False,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_dataset(dataset_name, data_dir, random_seed=42):
    """Load dataset with train/test split using dataset-specific configurations"""
    dataset_configs = get_dataset_configs(dataset_name)
    
    # Get dataset-specific parameters
    train_ratio = dataset_configs['ratio_config'].get('train_ratio', 0.7)
    temporal_split = dataset_configs['ratio_config'].get('temporal_split', False)
    
    print(f"\nLoading {dataset_configs['name']} dataset...")
    print(f"  Train ratio: {train_ratio}, Temporal split: {temporal_split}")
    
    # Calculate total samples needed
    total_train_samples = dataset_configs['config']['num_training'] + dataset_configs['config']['num_removes']
    import math
    
    # For MNIST and w8a: use direct sample counts (have separate test sets)
    # For other datasets: adjust for train/test split ratio
    if dataset_name in ['mnist', 'w8a']:
        total_samples = total_train_samples
    else:
        total_samples = math.ceil(total_train_samples / train_ratio)
    
    # Load dataset using the appropriate loader
    loader_kwargs = {
        'data_dir': data_dir,
        'max_samples': total_samples,
        'train_ratio': train_ratio,
        'temporal_split': temporal_split,
        'random_seed': random_seed,
    }
    
    # Add dataset-specific parameters
    if dataset_name == 'ct_slice':
        loader_kwargs['imbalanced'] = dataset_configs['imbalanced']
    elif dataset_name == 'mnist':
        loader_kwargs['num_base'] = dataset_configs['config']['num_training']
        if 'base_imbalance_ratio' in dataset_configs['config']:
            loader_kwargs['base_imbalance_ratio'] = dataset_configs['config']['base_imbalance_ratio']
        if 'incremental_imbalance_ratio' in dataset_configs['config']:
            loader_kwargs['incremental_imbalance_ratio'] = dataset_configs['config']['incremental_imbalance_ratio']
        if 'dominant_class' in dataset_configs['config']:
            loader_kwargs['dominant_class'] = dataset_configs['config']['dominant_class']
    elif dataset_name == 'w8a':
        # w8a doesn't use temporal_split parameter
        del loader_kwargs['temporal_split']
    
    X_train, X_test, y_train, y_test = dataset_configs['loader'](**loader_kwargs)
    
    print(f"  Training samples: {X_train.size(0)}")
    print(f"  Test samples: {X_test.size(0)}")
    print(f"  Feature dimension: {X_train.size(1)}")
    print(f"  Base training: {dataset_configs['config']['num_training']} samples")
    print(f"  Total update data: {dataset_configs['config']['num_removes']} samples")
    
    # Verify we have enough training samples
    if X_train.size(0) < total_train_samples:
        raise ValueError(f"Not enough training samples! Need {total_train_samples}, got {X_train.size(0)}")
    
    return X_train, X_test, y_train, y_test, dataset_configs


def run_incremental_with_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios):
    """
    Run incremental method once and record metrics at different checkpoints.
    Uses the same Newton update method as run_incremental_experiment.
    
    Returns:
        dict: {ratio: (total_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_loss, lr_grad, lr_hessian_inv, lr_optimize, spectral_norm
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    # Train initial model on base data (same as run_incremental_experiment)
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Initialize for incremental updates (same as run_incremental_experiment)
    K = X_train_base.t().mm(X_train_base)
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    start_all = time.time()
    
    # Perform incremental updates (same logic as run_incremental_experiment)
    for i in range(num_removes):
        # Get new data point index
        new_point_idx = num_training + i
        
        # Initialize data for first point
        if i == 0:
            X_inc = X_train[:num_training]  # base data only
            y_inc = y_train[:num_training]
            # Calculate initial Hessian inverse on base data only
            H_inv = lr_hessian_inv(w_approx, X_inc, y_inc, lam)
        
        # Add new incremental data
        X_inc = torch.cat([X_inc, X_train[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train[new_point_idx].unsqueeze(0)])
        
        # Calculate Hessian inverse on the incremental dataset
        H_inv = lr_hessian_inv(w_approx, X_inc, y_inc, lam)
        
        # Calculate gradient on the newly added data point
        grad_i = lr_grad(w_approx, X_train[new_point_idx].unsqueeze(0), y_train[new_point_idx].unsqueeze(0), lam)
        
        # Update covariance matrix K with new data point
        K += torch.ger(X_train[new_point_idx], X_train[new_point_idx])
        
        # Newton update
        Delta = H_inv.mv(grad_i)
        w_approx -= Delta
        
        # Check if this is a checkpoint
        if (i + 1) in checkpoint_indices:
            current_time = time.time() - start_all
            
            # Evaluate
            pred = X_test.mv(w_approx)
            test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
            num_samples = num_training + i + 1
            
            # Find which ratio this corresponds to
            ratio = checkpoint_ratios[checkpoint_indices.index(i + 1)]
            checkpoint_results[ratio] = (current_time, test_acc, num_samples)
    
    return checkpoint_results


def run_streaming_with_checkpoints(X_train, y_train, X_test, y_test, args, beta, lambda_reg, checkpoint_ratios):
    """
    Run streaming method once and record metrics at different checkpoints.
    Uses the same streaming logic as run_streaming_experiment.
    
    Returns:
        dict: {ratio: (total_time, test_accuracy, num_samples)}
    """
    from core_functions import (
        lr_loss, lr_grad, lr_hessian_inv, lr_hessian_inv_approximate, lr_optimize,
        spectral_norm, compute_online_leverage_score_incremental, update_XTX_inverse,
        compute_sampling_probability
    )
    import math
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    # Calculate epsilon (same as run_streaming_experiment)
    delta = 0.01
    epsilon = delta / lambda_reg
    
    torch.manual_seed(42)
    
    # Move to device
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    # Train initial model (same as run_streaming_experiment)
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Initialize for incremental updates (same as run_streaming_experiment)
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    update_count = 0
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    start_all = time.time()
    
    # Perform streaming updates (same logic as run_streaming_experiment)
    for i in range(num_removes):
        # Get new data point index
        new_point_idx = num_training + i
        
        # Initialize data for first point
        if i == 0:
            X_inc = X_train[:num_training]  # base data only
            y_inc = y_train[:num_training]
            # Sampled matrix starts empty (not including base training set)
            X_inc_sample = torch.empty(0, X_train.size(1)).float().to(device)
            y_inc_sample = torch.empty(0).to(device)
            # Calculate initial Hessian inverse on base data only
            H_inv = lr_hessian_inv(w_approx, X_inc, y_inc, lam)
            # Initialize Gram_sample_inverse (not including base training set)
            I = torch.eye(X_train.size(1)).float().to(device)
            Gram_sample_inverse = (lambda_reg * I).inverse()
        
        # Original incremental data
        X_inc = torch.cat([X_inc, X_train[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train[new_point_idx].unsqueeze(0)])
        
        # Calculate gradient on the newly added data point
        grad_i = lr_grad(w_approx, X_train[new_point_idx].unsqueeze(0), y_train[new_point_idx].unsqueeze(0), lam)
        Delta = grad_i
        
        # Calculate the sampling probability
        leverage_score = compute_online_leverage_score_incremental(
            X_train[new_point_idx], Gram_sample_inverse, X_inc_sample, epsilon, lambda_reg)
        sample_prob = compute_sampling_probability(leverage_score, d, beta, epsilon)
        
        # Sample point with probability sample_prob
        if torch.rand(1).item() < sample_prob:
            # Increment update counter
            update_count += 1
            
            # Add sampled point (with rescaling)
            weighted_x = X_train[new_point_idx] / math.sqrt(sample_prob)
            X_inc_sample = torch.cat([X_inc_sample, weighted_x.unsqueeze(0)])
            y_inc_sample = torch.cat([y_inc_sample, y_train[new_point_idx].unsqueeze(0)])
            
            # Calculate Hessian inverse
            X_for_hessian = torch.cat([X_train[:num_training], X_inc_sample])
            y_for_hessian = torch.cat([y_train[:num_training], y_inc_sample])
            H_inv = lr_hessian_inv_approximate(w_approx, X_for_hessian, X_inc, y_for_hessian, lam)
            
            # Update Gram_sample_inverse
            Gram_sample_inverse = update_XTX_inverse(weighted_x, Gram_sample_inverse, lambda_reg)
            
            # Newton update
            Delta_update = H_inv.mv(Delta)
            w_approx -= Delta_update
        
        # Check if this is a checkpoint
        if (i + 1) in checkpoint_indices:
            current_time = time.time() - start_all
            
            # Evaluate
            pred = X_test.mv(w_approx)
            test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
            num_samples = num_training + update_count
            
            ratio = checkpoint_ratios[checkpoint_indices.index(i + 1)]
            checkpoint_results[ratio] = (current_time, test_acc, num_samples)
    
    return checkpoint_results


def run_uniform_with_checkpoints(X_train, y_train, X_test, y_test, args, sample_prob, checkpoint_ratios):
    """
    Run uniform sampling method once and record metrics at different checkpoints.
    Uses the same uniform sampling logic as run_uniform_experiment.
    
    Returns:
        dict: {ratio: (total_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_loss, lr_grad, lr_hessian_inv, lr_hessian_inv_approximate, lr_optimize
    import math
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    # Train initial model (same as run_uniform_experiment)
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Initialize for incremental updates (same as run_uniform_experiment)
    K = X_train_base.t().mm(X_train_base)
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    update_count = 0
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    start_all = time.time()
    
    # Perform uniform sampling updates (same logic as run_uniform_experiment)
    for i in range(num_removes):
        # Get new data point index
        new_point_idx = num_training + i
        
        # Initialize data for first point
        if i == 0:
            X_inc = X_train[:num_training]  # base data only
            X_inc_sample = X_train[:num_training]
            y_inc = y_train[:num_training]
            y_inc_sample = y_train[:num_training]
            # Calculate initial Hessian inverse on base data only
            H_inv = lr_hessian_inv(w_approx, X_inc, y_inc, lam)
            # Initialize Gram_sample_inverse
            lambda_reg = 0.1
            Gram = X_inc_sample.t().mm(X_inc_sample)
            I = torch.eye(X_inc_sample.size(1)).to(device)
            Gram_sample_inverse = (Gram + lambda_reg * I).inverse()
        
        # Original incremental data
        X_inc = torch.cat([X_inc, X_train[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train[new_point_idx].unsqueeze(0)])
        
        # Calculate gradient on the newly added data point
        grad_i = lr_grad(w_approx, X_train[new_point_idx].unsqueeze(0), y_train[new_point_idx].unsqueeze(0), lam)
        Delta = grad_i
        
        # Sample point with fixed probability
        if torch.rand(1).item() < sample_prob:
            # Increment update counter
            update_count += 1
            
            # Add sampled point (with rescaling)
            X_inc_sample = torch.cat([X_inc_sample, X_train[new_point_idx].unsqueeze(0) / math.sqrt(sample_prob)])
            y_inc_sample = torch.cat([y_inc_sample, y_train[new_point_idx].unsqueeze(0)])
            
            # Calculate Hessian inverse on the sampled incremental dataset
            H_inv = lr_hessian_inv_approximate(w_approx, X_inc_sample, X_inc, y_inc_sample, lam)
            
            # Update covariance matrix K with new data point
            K += torch.ger(X_train[new_point_idx], X_train[new_point_idx])
            
            # Newton update
            delta_newton = H_inv.mv(Delta)
            w_approx -= delta_newton
            
            # Reset Delta to 0 vector
            Delta = torch.zeros_like(Delta)
        
        # Check if this is a checkpoint
        if (i + 1) in checkpoint_indices:
            current_time = time.time() - start_all
            
            # Evaluate
            pred = X_test.mv(w_approx)
            test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
            num_samples = len(X_inc_sample)
            
            ratio = checkpoint_ratios[checkpoint_indices.index(i + 1)]
            checkpoint_results[ratio] = (current_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios, tol=1e-10):
    """
    Retrain model from scratch at each checkpoint with all data up to that point.
    This serves as a baseline to compare against incremental methods.
    Uses torchmin L-BFGS implementation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
        tol: Tolerance for L-BFGS convergence (default: 1e-10)
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_lbfgs_torchmin
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using torchmin L-BFGS
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_lbfgs_torchmin(X_current, y_current, lam, b=b, num_steps=num_steps, tol=tol, verbose=False)
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_newton_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios, lr=1.0):
    """
    Retrain model from scratch at each checkpoint using Newton's method (NewtonExact).
    This serves as a baseline to compare against incremental methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
        lr: Learning rate for Newton's method (default: 1.0)
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_newton
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using Newton's method
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_newton(X_current, y_current, lam, b=b, num_steps=num_steps, lr=lr, verbose=False)
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_bfgs_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios, tol=1e-10):
    """
    Retrain model from scratch at each checkpoint using BFGS method.
    This serves as a baseline to compare against incremental methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
        tol: Tolerance for BFGS convergence (default: 1e-10)
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_bfgs
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using BFGS
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_bfgs(X_current, y_current, lam, b=b, num_steps=num_steps, tol=tol, verbose=False)
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_trust_exact_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios):
    """
    Retrain model from scratch at each checkpoint using Trust-Region Exact method.
    This serves as a baseline to compare against incremental methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_trust_exact
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using Trust-Region Exact
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_trust_exact(X_current, y_current, lam, b=b, num_steps=num_steps, verbose=False)
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_subsampled_newton_lev_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios, hessian_size=None, mh=10):
    """
    Retrain model from scratch at each checkpoint using Subsampled Newton with leverage score sampling.
    This serves as a baseline to compare against incremental methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
        hessian_size: Expected number of samples for Hessian approximation (default: 20*d)
        mh: Hessian approximation frequency (default: 10)
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_subsampled_newton_lev
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using Subsampled Newton (Leverage)
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_subsampled_newton_lev(
            X_current, y_current, lam, b=b, 
            num_steps=num_steps, 
            hessian_size=hessian_size, 
            mh=mh, 
            verbose=False
        )
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_retrain_subsampled_newton_uniform_at_checkpoints(X_train, y_train, X_test, y_test, args, checkpoint_ratios, hessian_size=None):
    """
    Retrain model from scratch at each checkpoint using Subsampled Newton with uniform sampling.
    This serves as a baseline to compare against incremental methods.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        args: Dictionary with experiment parameters
        checkpoint_ratios: List of data ratios to evaluate
        hessian_size: Number of samples for Hessian approximation (default: 20*d)
    
    Returns:
        dict: {ratio: (retrain_time, test_accuracy, num_samples)}
    """
    from core_functions import lr_optimize_subsampled_newton_uniform
    
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    torch.manual_seed(42)
    
    # Move to device
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    # Calculate checkpoint indices
    checkpoint_indices = [int(num_removes * ratio) for ratio in checkpoint_ratios]
    checkpoint_results = {}
    
    # Use the same perturbation vector for fair comparison
    b = std * torch.randn(X_train.size(1)).float().to(device)
    
    # For each checkpoint, retrain from scratch using Subsampled Newton (Uniform)
    for ratio in checkpoint_ratios:
        num_incremental = int(num_removes * ratio)
        
        # Prepare training data up to this checkpoint
        X_current = torch.cat([X_train_base, X_train[num_training:num_training + num_incremental]], dim=0)
        y_current = torch.cat([y_train_base, y_train[num_training:num_training + num_incremental]], dim=0)
        
        # Time the retraining
        start_time = time.time()
        w = lr_optimize_subsampled_newton_uniform(
            X_current, y_current, lam, b=b, 
            num_steps=num_steps, 
            hessian_size=hessian_size, 
            verbose=False
        )
        retrain_time = time.time() - start_time
        
        # Evaluate
        pred = X_test.mv(w)
        test_acc = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        num_samples = len(X_current)
        
        checkpoint_results[ratio] = (retrain_time, test_acc, num_samples)
    
    return checkpoint_results


def run_dataset_experiments(dataset_name, data_dir, num_repeats):
    """Run all experiments for a single dataset using dataset-specific configurations"""
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENTS FOR {dataset_name.upper()} DATASET")
    print("="*80)
    
    results = []
    
    # Get dataset configurations
    dataset_configs = get_dataset_configs(dataset_name)
    
    # Run experiments with multiple repetitions
    for repeat_id in range(1, num_repeats + 1):
        print(f"\n{'='*60}")
        print(f"REPETITION {repeat_id}/{num_repeats}")
        print(f"{'='*60}")
        
        # Load dataset with different random seed for each repetition
        random_seed = 42 + repeat_id - 1
        X_train, X_test, y_train, y_test, _ = load_dataset(
            dataset_name, data_dir, random_seed
        )
        
        # Prepare arguments using dataset-specific parameters
        exp_args = {
            'lam': dataset_configs['ratio_config'].get('lam', COMMON_ARGS['lam']),
            'std': dataset_configs['ratio_config'].get('std', COMMON_ARGS['std']),
            'num_training': dataset_configs['config']['num_training'],
            'num_removes': dataset_configs['config']['num_removes'],
            'num_steps': INCREMENTAL_RATIO_NUM_STEPS,
        }
        
        print(f"  Using lam={exp_args['lam']}, std={exp_args['std']}, num_steps={exp_args['num_steps']}")
        
        # Get streaming and uniform configurations for this dataset
        streaming_config = dataset_configs['ratio_config']['streaming']
        uniform_config = dataset_configs['ratio_config']['uniform']
        beta, lambda_reg = streaming_config
        sample_prob = uniform_config
        
        print(f"\nRunning experiments with checkpoint ratios: {[f'{r*100:.0f}%' for r in INCREMENTAL_DATA_RATIOS]}")
        
        # 0. Retrain from scratch at each checkpoint (baseline - L-BFGS)
        print(f"\n  Running Retrain (L-BFGS) from scratch at each checkpoint...")
        retrain_checkpoints = run_retrain_at_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args, INCREMENTAL_DATA_RATIOS
        )
        for ratio, (retrain_time, test_acc, num_samples) in retrain_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Retrain',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': retrain_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 0b. Retrain from scratch at each checkpoint (baseline - Newton)
        print(f"\n  Running Retrain (Newton) from scratch at each checkpoint...")
        retrain_newton_checkpoints = run_retrain_newton_at_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args, INCREMENTAL_DATA_RATIOS, lr=1.0
        )
        for ratio, (retrain_time, test_acc, num_samples) in retrain_newton_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Retrain_Newton',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': retrain_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 0c. Retrain from scratch at each checkpoint (baseline - BFGS)
        print(f"\n  Running Retrain (BFGS) from scratch at each checkpoint...")
        retrain_bfgs_checkpoints = run_retrain_bfgs_at_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args, INCREMENTAL_DATA_RATIOS, tol=1e-10
        )
        for ratio, (retrain_time, test_acc, num_samples) in retrain_bfgs_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Retrain_BFGS',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': retrain_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 0d. Retrain from scratch at each checkpoint (baseline - Trust-Region Exact)
        print(f"\n  Running Retrain (Trust-Region Exact) from scratch at each checkpoint...")
        retrain_trust_checkpoints = run_retrain_trust_exact_at_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args, INCREMENTAL_DATA_RATIOS
        )
        for ratio, (retrain_time, test_acc, num_samples) in retrain_trust_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Retrain_TrustExact',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': retrain_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 0e. Retrain from scratch at each checkpoint (baseline - Subsampled Newton Leverage)
        # COMMENTED OUT - Leverage method has accuracy issues
        # print(f"\n  Running Retrain (Subsampled Newton Lev) from scratch at each checkpoint...")
        # # Newton methods converge faster but need enough iterations for small lambda
        # sn_args = exp_args.copy()
        # if dataset_name == 'ct_slice':
        #     # CT Slice has very small lambda (1e-5), needs more iterations
        #     sn_args['num_steps'] = 200
        #     # Use 30% of data for stable Hessian with small lambda
        #     sn_hessian_size = min(int(0.3 * X_train.size(0)), 20 * X_train.size(1))
        # elif dataset_name == 'mnist':
        #     # MNIST has high dimension (784), allocate more samples
        #     sn_args['num_steps'] = 20
        #     # Use 10*d or 10% of data, whichever is smaller
        #     sn_hessian_size = min(10 * X_train.size(1), int(0.02 * X_train.size(0)))
        # else:
        #     sn_args['num_steps'] = 50
        #     sn_hessian_size = None  # Use default
        # 
        # retrain_sn_lev_checkpoints = run_retrain_subsampled_newton_lev_at_checkpoints(
        #     X_train.clone(), y_train.clone(), X_test, y_test, sn_args, INCREMENTAL_DATA_RATIOS, 
        #     hessian_size=sn_hessian_size, mh=10
        # )
        # for ratio, (retrain_time, test_acc, num_samples) in retrain_sn_lev_checkpoints.items():
        #     results.append({
        #         'dataset': dataset_name,
        #         'repeat_id': repeat_id,
        #         'method': 'Retrain_SubsampledNewton_Lev',
        #         'data_ratio': ratio,
        #         'beta': None,
        #         'lambda': None,
        #         'sample_prob': None,
        #         'total_time': retrain_time,
        #         'test_accuracy': test_acc,
        #         'num_samples': num_samples,
        #     })
        #     print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # Configuration for Subsampled Newton Uniform method
        sn_args = exp_args.copy()
        if dataset_name == 'ct_slice':
            # CT Slice has very small lambda (1e-5), needs more iterations
            sn_args['num_steps'] = 200
            # Use 30% of data for stable Hessian with small lambda
            sn_hessian_size = min(int(0.3 * X_train.size(0)), 20 * X_train.size(1))
        elif dataset_name == 'mnist':
            # MNIST has high dimension (784), allocate more samples
            sn_args['num_steps'] = 20
            # Use 10*d or 10% of data, whichever is smaller
            sn_hessian_size = min(10 * X_train.size(1), int(0.02 * X_train.size(0)))
        else:
            sn_args['num_steps'] = 50
            sn_hessian_size = None  # Use default
        
        # 0f. Retrain from scratch at each checkpoint (baseline - Subsampled Newton Uniform)
        print(f"\n  Running Retrain (Subsampled Newton Uniform) from scratch at each checkpoint...")
        # Use same configuration as leverage method for fair comparison
        retrain_sn_uniform_checkpoints = run_retrain_subsampled_newton_uniform_at_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, sn_args, INCREMENTAL_DATA_RATIOS,
            hessian_size=sn_hessian_size
        )
        for ratio, (retrain_time, test_acc, num_samples) in retrain_sn_uniform_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Retrain_SubsampledNewton_Uniform',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': retrain_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={retrain_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 1. Incremental method - run once and record at checkpoints
        print(f"\n  Running Incremental (full run with checkpoints)...")
        incremental_checkpoints = run_incremental_with_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args, INCREMENTAL_DATA_RATIOS
        )
        for ratio, (total_time, test_acc, num_samples) in incremental_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Incremental',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': None,
                'total_time': total_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={total_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 2. Streaming method - run once and record at checkpoints
        print(f"\n  Running Streaming (β={beta}, λ={lambda_reg}, full run with checkpoints)...")
        streaming_checkpoints = run_streaming_with_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test, 
            exp_args, beta, lambda_reg, INCREMENTAL_DATA_RATIOS
        )
        for ratio, (total_time, test_acc, num_samples) in streaming_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Streaming',
                'data_ratio': ratio,
                'beta': beta,
                'lambda': lambda_reg,
                'sample_prob': None,
                'total_time': total_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={total_time:.4f}s, Accuracy={test_acc:.4f}")
        
        # 3. Uniform method - run once and record at checkpoints
        print(f"\n  Running Uniform (p={sample_prob}, full run with checkpoints)...")
        uniform_checkpoints = run_uniform_with_checkpoints(
            X_train.clone(), y_train.clone(), X_test, y_test,
            exp_args, sample_prob, INCREMENTAL_DATA_RATIOS
        )
        for ratio, (total_time, test_acc, num_samples) in uniform_checkpoints.items():
            results.append({
                'dataset': dataset_name,
                'repeat_id': repeat_id,
                'method': 'Uniform',
                'data_ratio': ratio,
                'beta': None,
                'lambda': None,
                'sample_prob': sample_prob,
                'total_time': total_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
            print(f"    @ {ratio*100:.0f}%: Time={total_time:.4f}s, Accuracy={test_acc:.4f}")
    
    return results


def save_results(results, csv_path):
    """Save results to CSV file"""
    print(f"\nSaving results to {csv_path}...")
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'repeat_id', 'method', 'data_ratio', 'beta', 'lambda', 
            'sample_prob', 'total_time', 'test_accuracy', 'num_samples'
        ])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved successfully!")


def aggregate_results(results):
    """Aggregate results from multiple repetitions"""
    df = pd.DataFrame(results)
    
    # Group by dataset, method, and data_ratio
    grouped = df.groupby(['dataset', 'method', 'data_ratio']).agg({
        'total_time': ['mean', 'std'],
        'test_accuracy': ['mean', 'std'],
        'num_samples': ['mean', 'std'],
        'beta': 'first',
        'lambda': 'first',
        'sample_prob': 'first'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    return grouped


# Visualization function removed - only CSV output is needed
# def plot_results(results, plot_dir):
#     """Generate visualization plots for all datasets"""
#     pass


def print_summary_statistics(results):
    """Print summary statistics for all datasets"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    agg_results = aggregate_results(results)
    
    for dataset in agg_results['dataset'].unique():
        print(f"\n{dataset.upper()} Dataset:")
        print("-" * 60)
        dataset_data = agg_results[agg_results['dataset'] == dataset]
        
        for ratio in INCREMENTAL_DATA_RATIOS:
            ratio_data = dataset_data[dataset_data['data_ratio'] == ratio]
            if len(ratio_data) > 0:
                print(f"\n  {ratio*100:.0f}% Incremental Data:")
                for _, row in ratio_data.iterrows():
                    method = row['method']
                    time_mean = row['total_time_mean']
                    time_std = row['total_time_std']
                    acc_mean = row['test_accuracy_mean'] * 100
                    acc_std = row['test_accuracy_std'] * 100
                    print(f"    {method:12s}: Time={time_mean:7.4f}±{time_std:6.4f}s, "
                          f"Accuracy={acc_mean:6.2f}±{acc_std:5.2f}%")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Incremental Data Ratio Experiments for Multiple Datasets'
    )
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['mnist', 'ct_slice', 'dynamic_share', 'w8a'],
                       choices=['mnist', 'ct_slice', 'dynamic_share', 'w8a'],
                       help='Datasets to run experiments on (default: all four)')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Data directory (default: ../data)')
    parser.add_argument('--num-repeats', type=int, default=5,
                       help='Number of repeated experiments (default: 5)')
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("\n" + "="*80)
    print("INCREMENTAL DATA RATIO EXPERIMENTS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Data ratios: {[f'{r*100:.0f}%' for r in INCREMENTAL_DATA_RATIOS]}")
    print(f"  Methods: 6 Retrain baselines + 3 Incremental methods")
    print(f"  Number of repeats: {args.num_repeats}")
    print(f"  Optimization steps: {INCREMENTAL_RATIO_NUM_STEPS} (same for all datasets)")
    print(f"  Note: Each dataset uses its own lam, std, train_ratio, and split strategy (see configs.py)")
    print(f"\n  Method descriptions:")
    print(f"    Retrain Baselines (retrain from scratch at each checkpoint):")
    print(f"      - Retrain (L-BFGS): Standard L-BFGS optimization")
    print(f"      - Retrain (Newton): Exact Newton's method")
    print(f"      - Retrain (BFGS): Full BFGS optimization")
    print(f"      - Retrain (Trust-Exact): Trust-Region Exact method")
    # print(f"      - Retrain (Subsampled Newton Lev): Subsampled Newton with leverage score sampling")
    print(f"      - Retrain (Subsampled Newton Uniform): Subsampled Newton with uniform sampling")
    print(f"    Incremental Methods (update from previous model):")
    print(f"      - Incremental: Full incremental updates (no sampling)")
    print(f"      - Streaming: Leverage score sampling")
    print(f"      - Uniform: Uniform random sampling")
    
    # Collect results from all datasets
    all_results = []
    
    # Run experiments for each dataset
    for dataset_name in args.datasets:
        dataset_results = run_dataset_experiments(
            dataset_name, args.data_dir, args.num_repeats
        )
        all_results.extend(dataset_results)
    
    print("\n" + "="*80)
    print(f"ALL EXPERIMENTS COMPLETED!")
    print(f"Total experiments run: {len(all_results)}")
    print("="*80)
    
    # Save results
    output_dir = 'results/csv'
    csv_path = os.path.join(output_dir, 'incremental_ratio_comparison.csv')
    save_results(all_results, csv_path)
    
    # Save aggregated results
    agg_results = aggregate_results(all_results)
    agg_csv_path = os.path.join(output_dir, 'incremental_ratio_comparison_aggregated.csv')
    agg_results.to_csv(agg_csv_path, index=False)
    print(f"Aggregated results saved to: {agg_csv_path}")
    
    # Print summary statistics
    print_summary_statistics(all_results)
    
    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Raw results: {csv_path}")
    print(f"Aggregated results: {agg_csv_path}")


if __name__ == '__main__':
    main()

