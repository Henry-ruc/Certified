# Main comparison script for Pareto frontier analysis - Version 2.0
# Compares three methods: Incremental (full), Streaming (leverage), and Uniform sampling
# V2.0: Supports multiple datasets (CT Slice, Dynamic Share)

from operator import truediv
import sys
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import experiment runners and utilities
from run_incremental import run_incremental_experiment
from run_streaming import run_streaming_experiment
from run_uniform import run_uniform_experiment
from data_loader import (
    load_ctslice_for_experiments, 
    load_dynamic_share_for_experiments, 
    load_mnist_for_experiments,
    load_covertype_for_experiments,
    load_a1a_for_experiments,
    load_w8a_for_experiments
)
from configs import (
    DATASET_CONFIG, DATASET_CONFIG_DYNAMIC, DATASET_CONFIG_MNIST, DATASET_CONFIG_COVERTYPE, DATASET_CONFIG_A1A, DATASET_CONFIG_W8A,
    COMMON_ARGS, STREAMING_CONFIGS, UNIFORM_CONFIGS,
    STREAMING_CONFIGS_DYNAMIC, UNIFORM_CONFIGS_DYNAMIC,
    STREAMING_CONFIGS_MNIST, UNIFORM_CONFIGS_MNIST,
    STREAMING_CONFIGS_COVERTYPE, UNIFORM_CONFIGS_COVERTYPE,
    STREAMING_CONFIGS_A1A, UNIFORM_CONFIGS_A1A,
    STREAMING_CONFIGS_W8A, UNIFORM_CONFIGS_W8A,
    OUTPUT_CONFIG, OUTPUT_CONFIG_DYNAMIC, OUTPUT_CONFIG_MNIST, OUTPUT_CONFIG_COVERTYPE, OUTPUT_CONFIG_A1A, OUTPUT_CONFIG_W8A
)
from core_functions import lr_optimize, device
import time


def compute_baselines(X_train, y_train, X_test, y_test, args):
    """
    Compute two baselines:
    1. Base model: test accuracy of model trained only on base training set
    2. Retrain: test accuracy and training time of model retrained on all data (base + updates)
    
    Returns:
        tuple: (base_acc, retrain_acc, retrain_time)
    """
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
    
    # 1. Base model: trained only on base training set
    print(f"  Training base model on {num_training} samples...")
    X_train_base = X_train[:num_training]
    y_train_base = y_train[:num_training]
    
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w_base = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=True)
    
    # Evaluate base model
    pred_base = X_test.mv(w_base)
    base_acc = pred_base.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    print(f"    Test accuracy: {base_acc:.4f}")
    
    # 2. Retrain: retrain on all data
    total_samples = num_training + num_removes
    print(f"  Retraining on all {total_samples} samples...")
    X_train_all = X_train[:total_samples]
    y_train_all = y_train[:total_samples]
    
    b_retrain = std * torch.randn(X_train_all.size(1)).float().to(device)
    
    start_time = time.time()
    w_retrain = lr_optimize(X_train_all, y_train_all, lam, b=b_retrain, num_steps=num_steps, verbose=True)
    retrain_time = time.time() - start_time
    
    # Evaluate retrained model
    pred_retrain = X_test.mv(w_retrain)
    retrain_acc = pred_retrain.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    print(f"    Test accuracy: {retrain_acc:.4f}")
    print(f"    Training time: {retrain_time:.2f}s")
    
    return base_acc, retrain_acc, retrain_time


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Compare incremental learning methods using Pareto frontier (V2.0)')
    parser.add_argument('--dataset', type=str, default='ct_slice', 
                       choices=['ct_slice', 'dynamic_share', 'mnist', 'covertype', 'a1a', 'w8a'],
                       help='Dataset to use (default: ct_slice)')
    parser.add_argument('--data-dir', type=str, default='../data', 
                       help='Data directory (default: ../data)')
    parser.add_argument('--skip-incremental', action='store_true', 
                       help='Skip incremental baseline (if already run)')
    parser.add_argument('--skip-streaming', action='store_true', 
                       help='Skip streaming experiments')
    parser.add_argument('--skip-uniform', action='store_true', 
                       help='Skip uniform experiments')
    parser.add_argument('--num-repeats', type=int, default=5, 
                       help='Number of repeated experiments (default: 5)')
    parser.add_argument('--train-ratio', type=float, default=0.7, 
                       help='Training data ratio (default: 0.7)')
    parser.add_argument('--temporal-split', action='store_true',
                       help='Use temporal split (default: random split)')
    return parser.parse_args()


def get_dataset_configs(args):
    """Get dataset-specific configurations"""
    if args.dataset == 'ct_slice':
        return {
            'config': DATASET_CONFIG,
            'streaming_configs': STREAMING_CONFIGS,
            'uniform_configs': UNIFORM_CONFIGS,
            'output_config': OUTPUT_CONFIG,
            'loader': load_ctslice_for_experiments,
            'name': 'CT Slice',
            'max_samples': DATASET_CONFIG['num_training'] + DATASET_CONFIG['num_removes'],
            'imbalanced': True,
        }
    elif args.dataset == 'dynamic_share':
        return {
            'config': DATASET_CONFIG_DYNAMIC,
            'streaming_configs': STREAMING_CONFIGS_DYNAMIC,
            'uniform_configs': UNIFORM_CONFIGS_DYNAMIC,
            'output_config': OUTPUT_CONFIG_DYNAMIC,
            'loader': load_dynamic_share_for_experiments,
            'name': 'Dynamic Share',
            'max_samples': DATASET_CONFIG_DYNAMIC['num_training'] + DATASET_CONFIG_DYNAMIC['num_removes'],
            'imbalanced': False,
        }
    elif args.dataset == 'mnist':
        return {
            'config': DATASET_CONFIG_MNIST,
            'streaming_configs': STREAMING_CONFIGS_MNIST,
            'uniform_configs': UNIFORM_CONFIGS_MNIST,
            'output_config': OUTPUT_CONFIG_MNIST,
            'loader': load_mnist_for_experiments,
            'name': 'MNIST (Classes 3 and 8)',
            'max_samples': DATASET_CONFIG_MNIST['num_training'] + DATASET_CONFIG_MNIST['num_removes'],
            'imbalanced': False,
        }
    elif args.dataset == 'covertype':
        return {
            'config': DATASET_CONFIG_COVERTYPE,
            'streaming_configs': STREAMING_CONFIGS_COVERTYPE,
            'uniform_configs': UNIFORM_CONFIGS_COVERTYPE,
            'output_config': OUTPUT_CONFIG_COVERTYPE,
            'loader': load_covertype_for_experiments,
            'name': 'Covertype (Forest Cover Type)',
            'max_samples': DATASET_CONFIG_COVERTYPE['num_training'] + DATASET_CONFIG_COVERTYPE['num_removes'],
            'imbalanced': False,
        }
    elif args.dataset == 'a1a':
        return {
            'config': DATASET_CONFIG_A1A,
            'streaming_configs': STREAMING_CONFIGS_A1A,
            'uniform_configs': UNIFORM_CONFIGS_A1A,
            'output_config': OUTPUT_CONFIG_A1A,
            'loader': load_a1a_for_experiments,
            'name': 'a1a (Adult Dataset)',
            'max_samples': DATASET_CONFIG_A1A['num_training'] + DATASET_CONFIG_A1A['num_removes'],
            'imbalanced': False,
        }
    elif args.dataset == 'w8a':
        return {
            'config': DATASET_CONFIG_W8A,
            'streaming_configs': STREAMING_CONFIGS_W8A,
            'uniform_configs': UNIFORM_CONFIGS_W8A,
            'output_config': OUTPUT_CONFIG_W8A,
            'loader': load_w8a_for_experiments,
            'name': 'w8a Dataset',
            'max_samples': DATASET_CONFIG_W8A['num_training'] + DATASET_CONFIG_W8A['num_removes'],
            'imbalanced': False,
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def load_dataset(args, dataset_configs, random_seed=42):
    """Load dataset with train/test split"""
    print("="*60)
    print(f"Loading {dataset_configs['name']} dataset...")
    print("="*60)
    
    # Calculate total samples needed
    total_train_samples = dataset_configs['config']['num_training'] + dataset_configs['config']['num_removes']
    import math
    
    # For MNIST, a1a, w8a: use direct sample counts (have separate test sets)
    # For other datasets: adjust for train/test split ratio
    if args.dataset in ['mnist', 'a1a', 'w8a']:
        total_samples = total_train_samples
    else:
        total_samples = math.ceil(total_train_samples / args.train_ratio)
    
    # Load dataset using the appropriate loader
    loader_kwargs = {
        'data_dir': args.data_dir,
        'max_samples': total_samples,
        'train_ratio': args.train_ratio,
        'temporal_split': args.temporal_split,
        'random_seed': random_seed,
    }
    
    # Add dataset-specific parameters
    if args.dataset == 'ct_slice':
        loader_kwargs['imbalanced'] = dataset_configs['imbalanced']
    elif args.dataset == 'mnist':
        # Add MNIST-specific parameters
        loader_kwargs['num_base'] = dataset_configs['config']['num_training']
        if 'base_imbalance_ratio' in dataset_configs['config']:
            loader_kwargs['base_imbalance_ratio'] = dataset_configs['config']['base_imbalance_ratio']
        if 'incremental_imbalance_ratio' in dataset_configs['config']:
            loader_kwargs['incremental_imbalance_ratio'] = dataset_configs['config']['incremental_imbalance_ratio']
        if 'dominant_class' in dataset_configs['config']:
            loader_kwargs['dominant_class'] = dataset_configs['config']['dominant_class']
    elif args.dataset == 'covertype':
        # Add Covertype-specific parameters
        if 'binary_task' in dataset_configs['config']:
            loader_kwargs['binary_task'] = dataset_configs['config']['binary_task']
    elif args.dataset == 'a1a':
        # a1a doesn't use temporal_split parameter
        del loader_kwargs['temporal_split']
    elif args.dataset == 'w8a':
        # w8a doesn't use temporal_split parameter (uses original train/test split)
        del loader_kwargs['temporal_split']
    
    X_train, X_test, y_train, y_test = dataset_configs['loader'](**loader_kwargs)
    
    split_type = "Temporal" if args.temporal_split else "Random"
    print(f"\n{dataset_configs['name']} dataset loaded successfully!")
    print(f"  Training samples: {X_train.size(0)}")
    print(f"  Test samples: {X_test.size(0)}")
    print(f"  Feature dimension: {X_train.size(1)}")
    print(f"  Train/Test ratio: {args.train_ratio:.1f}/{1-args.train_ratio:.1f}")
    print(f"  Split strategy: {split_type}")
    print(f"  Base training: {dataset_configs['config']['num_training']} samples")
    print(f"  Update data: {dataset_configs['config']['num_removes']} samples")
    print(f"  Required train samples: {total_train_samples}")
    
    # Verify we have enough training samples
    if X_train.size(0) < total_train_samples:
        raise ValueError(f"Not enough training samples! Need {total_train_samples}, got {X_train.size(0)}")
    
    return X_train, X_test, y_train, y_test


def run_experiments(X_train, y_train, X_test, y_test, args, dataset_configs, repeat_id=1):
    """Run all experiments and collect results for one repetition"""
    results = []
    
    print(f"\n{'='*60}")
    print(f"REPEAT {repeat_id}/{args.num_repeats}")
    print(f"{'='*60}")
    
    # Prepare common arguments
    exp_args = {
        'lam': COMMON_ARGS['lam'],
        'std': COMMON_ARGS['std'],
        'num_training': dataset_configs['config']['num_training'],
        'num_removes': dataset_configs['config']['num_removes'],
        'num_steps': COMMON_ARGS['num_steps'],
    }
    
    # 0. Compute base model and full retrain baselines
    print("\n" + "="*60)
    print("BASELINE: Base Model and Full Retrain")
    print("="*60)
    base_acc, retrain_acc, retrain_time = compute_baselines(
        X_train.clone(), y_train.clone(), X_test, y_test, exp_args
    )
    
    # Add base model result (time=0, as it's already trained before updates)
    results.append({
        'repeat_id': repeat_id,
        'method': 'Base',
        'config': 'Base only',
        'beta': None,
        'lambda': None,
        'sample_prob': None,
        'total_time': 0.0,  # Base model is already trained
        'test_accuracy': base_acc,
        'num_samples': dataset_configs['config']['num_training'],
    })
    
    # Add full retrain result
    results.append({
        'repeat_id': repeat_id,
        'method': 'Retrain',
        'config': 'All data',
        'beta': None,
        'lambda': None,
        'sample_prob': None,
        'total_time': retrain_time,
        'test_accuracy': retrain_acc,
        'num_samples': dataset_configs['config']['num_training'] + dataset_configs['config']['num_removes'],
    })
    
    # 1. Run Incremental baseline
    if not args.skip_incremental:
        print("\n" + "="*60)
        print("EXPERIMENT 1/3: Incremental (Full Updates)")
        print("="*60)
        total_time, test_acc, num_samples = run_incremental_experiment(
            X_train.clone(), y_train.clone(), X_test, y_test, exp_args
        )
        results.append({
            'repeat_id': repeat_id,
            'method': 'Incremental',
            'config': 'Full',
            'beta': None,
            'lambda': None,
            'sample_prob': None,
            'total_time': total_time,
            'test_accuracy': test_acc,
            'num_samples': num_samples,
        })
    
    # 2. Run Streaming experiments
    if not args.skip_streaming:
        streaming_configs = dataset_configs['streaming_configs']
        print("\n" + "="*60)
        print(f"EXPERIMENT 2/3: Streaming (Leverage Sampling)")
        print(f"Running {len(streaming_configs)} configurations...")
        print("="*60)
        for idx, (beta, lambda_reg) in enumerate(streaming_configs, 1):
            print(f"\n[{idx}/{len(streaming_configs)}] Streaming with β={beta}, λ={lambda_reg}")
            total_time, test_acc, num_samples = run_streaming_experiment(
                X_train.clone(), y_train.clone(), X_test, y_test, 
                exp_args, beta, lambda_reg
            )
            results.append({
                'repeat_id': repeat_id,
                'method': 'Streaming',
                'config': f'β={beta},λ={lambda_reg}',
                'beta': beta,
                'lambda': lambda_reg,
                'sample_prob': None,
                'total_time': total_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
    
    # 3. Run Uniform experiments
    if not args.skip_uniform:
        uniform_configs = dataset_configs['uniform_configs']
        print("\n" + "="*60)
        print(f"EXPERIMENT 3/3: Uniform Sampling")
        print(f"Running {len(uniform_configs)} configurations...")
        print("="*60)
        for idx, sample_prob in enumerate(uniform_configs, 1):
            print(f"\n[{idx}/{len(uniform_configs)}] Uniform with prob={sample_prob}")
            total_time, test_acc, num_samples = run_uniform_experiment(
                X_train.clone(), y_train.clone(), X_test, y_test,
                exp_args, sample_prob
            )
            results.append({
                'repeat_id': repeat_id,
                'method': 'Uniform',
                'config': f'p={sample_prob}',
                'beta': None,
                'lambda': None,
                'sample_prob': sample_prob,
                'total_time': total_time,
                'test_accuracy': test_acc,
                'num_samples': num_samples,
            })
    
    return results


def save_results(results, csv_path):
    """Save results to CSV file"""
    print("\n" + "="*60)
    print(f"Saving results to {csv_path}")
    print("="*60)
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'repeat_id', 'method', 'config', 'beta', 'lambda', 'sample_prob',
            'total_time', 'test_accuracy', 'num_samples'
        ])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Results saved successfully!")


def aggregate_results(results):
    """Aggregate results from multiple repetitions"""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Group by method and config, compute mean and std
    grouped = df.groupby(['method', 'config']).agg({
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


def plot_pareto_frontier(results, plot_path, dataset_name, dataset_config, split_type):
    """Generate Pareto frontier plot with error bars from multiple repetitions"""
    print("\n" + "="*60)
    print(f"Generating Pareto frontier plot: {plot_path}")
    print("="*60)
    
    # Aggregate results (mean and std)
    agg_results = aggregate_results(results)
    
    # Separate results by method
    base_results = agg_results[agg_results['method'] == 'Base']
    retrain_results = agg_results[agg_results['method'] == 'Retrain'].copy()
    incremental_results = agg_results[agg_results['method'] == 'Incremental']
    streaming_results = agg_results[agg_results['method'] == 'Streaming']
    uniform_results = agg_results[agg_results['method'] == 'Uniform']
    
    # Adjust Retrain time by multiplying with (num_removes / 10)
    if len(retrain_results) > 0:
        time_multiplier = dataset_config['num_removes'] / 10
        retrain_results['total_time_mean'] = retrain_results['total_time_mean'] * time_multiplier
        retrain_results['total_time_std'] = retrain_results['total_time_std'] * time_multiplier
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Skip Base model and Retrain in Pareto frontier plot
    # Only show: Incremental, Streaming, Uniform
    
    # Plot Incremental (single point with marker only)
    if len(incremental_results) > 0:
        plt.plot(incremental_results['total_time_mean'].values, 
                incremental_results['test_accuracy_mean'].values * 100,
                '*', color='#E63946', markersize=22, 
                label='Incremental (Full)', zorder=5, 
                markeredgecolor='#8B0000', markeredgewidth=2.5)
    
    # Plot Streaming (line only, no markers) - Green
    if len(streaming_results) > 0:
        # Sort by time for line plot
        sorted_idx = streaming_results['total_time_mean'].argsort()
        sorted_times = streaming_results['total_time_mean'].values[sorted_idx]
        sorted_accs = streaming_results['test_accuracy_mean'].values[sorted_idx] * 100
        
        plt.plot(sorted_times, sorted_accs, '-', color='#388E3C', 
                linewidth=4.5, label='Streaming (Leverage)', zorder=3, alpha=0.95)
    
    # Plot Uniform (line only, no markers) - Blue
    if len(uniform_results) > 0:
        # Sort by time for line plot
        sorted_idx = uniform_results['total_time_mean'].argsort()
        sorted_times = uniform_results['total_time_mean'].values[sorted_idx]
        sorted_accs = uniform_results['test_accuracy_mean'].values[sorted_idx] * 100
        
        plt.plot(sorted_times, sorted_accs, '-', color='#1976D2', 
                linewidth=4.5, label='Uniform Sampling', zorder=3, alpha=0.95)
    
    plt.xlabel('Total Update Time (s)', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    
    title = f'Pareto Frontier: Update Time vs Test Accuracy\n{dataset_name} Dataset - {split_type} Split (Base: {dataset_config["num_training"]}, Updates: {dataset_config["num_removes"]})'
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Enhanced legend - lower right corner, larger font
    plt.legend(fontsize=18, loc='lower right', framealpha=0.95, 
              edgecolor='black', fancybox=True, shadow=True, frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # Save PNG
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as PNG: {plot_path}")
    # Save PDF
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as PDF: {pdf_path}")
    
    # Also show statistics
    print("\nSummary Statistics (mean ± std):")
    print("-" * 60)
    
    if len(base_results) > 0:
        row = base_results.iloc[0]
        print(f"Base Model: Accuracy={row['test_accuracy_mean']*100:.2f}±{row['test_accuracy_std']*100:.2f}% (no updates)")
    
    if len(retrain_results) > 0:
        row = retrain_results.iloc[0]
        print(f"Retrain (all data): Time={row['total_time_mean']:.2f}±{row['total_time_std']:.2f}s, "
              f"Accuracy={row['test_accuracy_mean']*100:.2f}±{row['test_accuracy_std']*100:.2f}%")
    
    if len(incremental_results) > 0:
        row = incremental_results.iloc[0]
        print(f"Incremental: Time={row['total_time_mean']:.2f}±{row['total_time_std']:.2f}s, "
              f"Accuracy={row['test_accuracy_mean']*100:.2f}±{row['test_accuracy_std']*100:.2f}%")
    
    if len(streaming_results) > 0:
        best_idx = streaming_results['test_accuracy_mean'].idxmax()
        fastest_idx = streaming_results['total_time_mean'].idxmin()
        best_stream = streaming_results.loc[best_idx]
        fastest_stream = streaming_results.loc[fastest_idx]
        print(f"Streaming: Best accuracy={best_stream['test_accuracy_mean']*100:.2f}±{best_stream['test_accuracy_std']*100:.2f}% ({best_stream['config']})")
        print(f"           Fastest={fastest_stream['total_time_mean']:.2f}±{fastest_stream['total_time_std']:.2f}s ({fastest_stream['config']})")
    
    if len(uniform_results) > 0:
        best_idx = uniform_results['test_accuracy_mean'].idxmax()
        fastest_idx = uniform_results['total_time_mean'].idxmin()
        best_uniform = uniform_results.loc[best_idx]
        fastest_uniform = uniform_results.loc[fastest_idx]
        print(f"Uniform: Best accuracy={best_uniform['test_accuracy_mean']*100:.2f}±{best_uniform['test_accuracy_std']*100:.2f}% ({best_uniform['config']})")
        print(f"         Fastest={fastest_uniform['total_time_mean']:.2f}±{fastest_uniform['total_time_std']:.2f}s ({fastest_uniform['config']})")


def plot_accuracy_vs_samples(results, plot_path, dataset_name, dataset_config, split_type):
    """Generate accuracy vs sample count plot with error bars from multiple repetitions"""
    print("\n" + "="*60)
    print(f"Generating accuracy vs samples plot: {plot_path}")
    print("="*60)
    
    # Aggregate results (mean and std)
    agg_results = aggregate_results(results)
    
    # Replace NaN std with 0 (happens when only 1 repetition)
    agg_results = agg_results.fillna(0)
    
    # Separate results by method
    base_results = agg_results[agg_results['method'] == 'Base']
    retrain_results = agg_results[agg_results['method'] == 'Retrain']
    incremental_results = agg_results[agg_results['method'] == 'Incremental']
    streaming_results = agg_results[agg_results['method'] == 'Streaming']
    uniform_results = agg_results[agg_results['method'] == 'Uniform']
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Plot Base model (single point with marker only)
    if len(base_results) > 0:
        plt.plot(base_results['num_samples_mean'].values, 
                base_results['test_accuracy_mean'].values * 100,
                'o', color='#6C757D', markersize=14, 
                label='Base Model (no updates)', zorder=5, 
                markeredgecolor='#495057', markeredgewidth=2.5)
    
    # Plot Retrain model (single point with marker only)
    if len(retrain_results) > 0:
        plt.plot(retrain_results['num_samples_mean'].values, 
                retrain_results['test_accuracy_mean'].values * 100,
                'D', color='#9D4EDD', markersize=14, 
                label='Retrain (all data)', zorder=5, 
                markeredgecolor='#6A1B9A', markeredgewidth=2.5)
    
    # Plot Incremental (single point with marker only)
    if len(incremental_results) > 0:
        plt.plot(incremental_results['num_samples_mean'].values, 
                incremental_results['test_accuracy_mean'].values * 100,
                '*', color='#E63946', markersize=22, 
                label='Incremental (Full)', zorder=5, 
                markeredgecolor='#8B0000', markeredgewidth=2.5)
    
    # Plot Streaming (line only, no markers) - Green
    if len(streaming_results) > 0:
        # Sort by num_samples for line plot
        sorted_idx = streaming_results['num_samples_mean'].argsort()
        sorted_samples = streaming_results['num_samples_mean'].values[sorted_idx]
        sorted_accs = streaming_results['test_accuracy_mean'].values[sorted_idx] * 100
        
        plt.plot(sorted_samples, sorted_accs, '-', color='#388E3C', 
                linewidth=4.5, label='Streaming (Leverage)', zorder=3, alpha=0.95)
    
    # Plot Uniform (line only, no markers) - Blue
    if len(uniform_results) > 0:
        # Sort by num_samples for line plot
        sorted_idx = uniform_results['num_samples_mean'].argsort()
        sorted_samples = uniform_results['num_samples_mean'].values[sorted_idx]
        sorted_accs = uniform_results['test_accuracy_mean'].values[sorted_idx] * 100
        
        plt.plot(sorted_samples, sorted_accs, '-', color='#1976D2', 
                linewidth=4.5, label='Uniform Sampling', zorder=3, alpha=0.95)
    
    plt.xlabel('Total Training Samples', fontsize=16, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=16, fontweight='bold')
    
    title = f'Test Accuracy vs Sample Count\n{dataset_name} Dataset - {split_type} Split (Base: {dataset_config["num_training"]}, Updates: {dataset_config["num_removes"]})'
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    
    # Enhanced legend - lower right corner, larger font
    plt.legend(fontsize=18, loc='lower right', framealpha=0.95, 
              edgecolor='black', fancybox=True, shadow=True, frameon=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot in both PNG and PDF formats
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    # Save PNG
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy vs samples plot saved as PNG: {plot_path}")
    # Save PDF
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Accuracy vs samples plot saved as PDF: {pdf_path}")


def main():
    """Main execution function with multiple repetitions"""
    args = parse_arguments()
    
    # Get dataset-specific configurations
    dataset_configs = get_dataset_configs(args)
    
    split_type = "Temporal" if args.temporal_split else "Random"
    
    print("\n" + "="*60)
    print(f"PARETO FRONTIER COMPARISON - {dataset_configs['name'].upper()} DATASET (V2.0)")
    print("="*60)
    print(f"Configuration:")
    print(f"  Dataset: {dataset_configs['name']}")
    print(f"  Base training samples: {dataset_configs['config']['num_training']}")
    print(f"  Update samples: {dataset_configs['config']['num_removes']}")
    print(f"  Train/Test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%} ({split_type})")
    print(f"  Number of repeats: {args.num_repeats}")
    print(f"  Streaming configs: {len(dataset_configs['streaming_configs'])}")
    print(f"  Uniform configs: {len(dataset_configs['uniform_configs'])}")
    total_exp = 1 + len(dataset_configs['streaming_configs']) + len(dataset_configs['uniform_configs'])
    print(f"  Total experiments: {args.num_repeats} × {total_exp} = {args.num_repeats * total_exp}")
    
    # Collect results from all repetitions
    all_results = []
    
    # Run experiments multiple times with different seeds
    for repeat_id in range(1, args.num_repeats + 1):
        print("\n" + "="*80)
        print(f"STARTING REPETITION {repeat_id}/{args.num_repeats}")
        print("="*80)
        
        # Load dataset with different random seed for each repetition
        random_seed = 42 + repeat_id - 1  # Seeds: 42, 43, 44, 45, 46
        X_train, X_test, y_train, y_test = load_dataset(args, dataset_configs, random_seed=random_seed)
        
        # Run experiments for this repetition
        results = run_experiments(X_train, y_train, X_test, y_test, args, dataset_configs, repeat_id=repeat_id)
        all_results.extend(results)
        
        print(f"\n✓ Repetition {repeat_id} completed: {len(results)} experiments")
    
    print("\n" + "="*80)
    print(f"ALL {args.num_repeats} REPETITIONS COMPLETED!")
    print(f"Total experiments run: {len(all_results)}")
    print("="*80)
    
    # Save all raw results
    output_config = dataset_configs['output_config']
    save_results(all_results, output_config['csv_path'])
    
    # Save aggregated results
    agg_results = aggregate_results(all_results)
    agg_csv_path = output_config['csv_path'].replace('.csv', '_aggregated.csv')
    agg_results.to_csv(agg_csv_path, index=False)
    print(f"Aggregated results saved to: {agg_csv_path}")
    
    # Generate Pareto frontier plot
    plot_pareto_frontier(all_results, output_config['plot_path'], 
                        dataset_configs['name'], dataset_configs['config'], split_type)
    
    # Generate accuracy vs samples plot
    samples_plot_path = output_config['plot_path'].replace('.png', '_samples.png')
    plot_accuracy_vs_samples(all_results, samples_plot_path,
                            dataset_configs['name'], dataset_configs['config'], split_type)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Raw results saved to: {output_config['csv_path']}")
    print(f"Aggregated results saved to: {agg_csv_path}")
    print(f"Pareto frontier plot saved to: {output_config['plot_path']}")
    print(f"Accuracy vs samples plot saved to: {samples_plot_path}")


if __name__ == '__main__':
    main()
