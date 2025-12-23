# Gradient Norm Tracking Experiment
# Runs incremental learning with gradient norm tracking and plots the results
# 
# NOTE: This experiment uses SEPARATE configurations from Pareto frontier experiments:
#   - GRADIENT_NORM_DATASET_CONFIG (vs DATASET_CONFIG)
#   - GRADIENT_NORM_ARGS (vs COMMON_ARGS)
# Results are saved in dedicated subdirectories:
#   - results/csv/gradient_norm/
#   - results/plots/gradient_norm/

import sys
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

# Import experiment runners and utilities
from run_incremental import run_incremental_with_grad_norm
from run_uniform import run_uniform_with_grad_norm
from run_streaming import run_streaming_with_grad_norm
from data_loader import load_ctslice_for_experiments, load_dynamic_share_for_experiments, load_mnist_for_experiments
from configs import (
    GRADIENT_NORM_DATASET_CONFIG, 
    GRADIENT_NORM_DATASET_CONFIG_DYNAMIC, 
    GRADIENT_NORM_DATASET_CONFIG_MNIST,
    GRADIENT_NORM_ARGS,
    GRADIENT_NORM_UNIFORM_PROB,
    GRADIENT_NORM_LEVERAGE_PARAMS
)

def plot_gradient_norm_results(all_results, dataset_name, output_dir):
    """
    Plot gradient residual norm vs incremental data size for multiple methods.
    Each method shows 3 lines: True, Data-Dependent Bound, Worst-Case Bound.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'method': str, 'results': dict, 'label': str}
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot
    """
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Color scheme: different colors for different methods
    method_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
    
    # Line styles for the three types
    line_styles = {
        'true': '-',           
        'data_dependent': '--',  
        'worst_case': ':'     
    }
    
    # Markers
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    # Plot each method
    for idx, result_dict in enumerate(all_results):
        method_name = result_dict['method']
        results = result_dict['results']
        label_prefix = result_dict.get('label', method_name)
        
        # Extract data
        grad_norm_approx = results['grad_norm_approx'].cpu().numpy()
        worst_case_bound = results['worst_case_bound_approx'].cpu().numpy()
        true_grad_norm_per_iter = results['true_grad_norm_per_iter'].cpu().numpy()
        num_iterations = len(grad_norm_approx)
        
        # Create incremental data indices
        incremental_indices = np.arange(1, num_iterations + 1)
        
        # Compute cumulative gradient norms
        cumulative_true = np.cumsum(true_grad_norm_per_iter)
        cumulative_data_dependent = np.cumsum(grad_norm_approx)
        cumulative_worst_case = np.cumsum(worst_case_bound)
        
        # Get base color and marker for this method
        base_color = method_colors[idx % len(method_colors)]
        marker = markers[idx % len(markers)]
        markevery = max(1, num_iterations // 20)
        
        # Convert hex color to RGB and create variations (light to dark)
        import matplotlib.colors as mcolors
        rgb = mcolors.hex2color(base_color)
        
        # Create three shades: light (True), medium (Data-Dependent), dark (Worst-Case)
        def adjust_lightness(color, factor):
            """Adjust color lightness. factor > 1 makes it lighter, < 1 makes it darker"""
            r, g, b = color
            # Convert to HSL-like lightness adjustment
            r = min(1, r + (1 - r) * (factor - 1) if factor > 1 else r * factor)
            g = min(1, g + (1 - g) * (factor - 1) if factor > 1 else g * factor)
            b = min(1, b + (1 - b) * (factor - 1) if factor > 1 else b * factor)
            return (r, g, b)
        
        color_light = adjust_lightness(rgb, 1.3)    # Lighter for True
        color_medium = rgb                           # Original for Data-Dependent  
        color_dark = adjust_lightness(rgb, 0.7)     # Darker for Worst-Case
        
        # Plot 3 lines for this method
        # 1. True gradient norm (solid line, lightest color)
        plt.semilogy(incremental_indices, cumulative_true, 
                     label=f'{label_prefix} - True', 
                     linestyle=line_styles['true'],
                     marker=marker, markersize=3, markevery=markevery,
                     linewidth=2.5, alpha=0.9, color=color_light)
        
        # 2. Data-dependent bound (dash-dot line, medium color)
        plt.semilogy(incremental_indices, cumulative_data_dependent, 
                     label=f'{label_prefix} - Data-Dependent', 
                     linestyle=line_styles['data_dependent'],
                     marker=marker, markersize=2, markevery=markevery,
                     linewidth=2, alpha=0.9, color=color_medium)
        
        # 3. Worst-case bound (dashed line, darkest color)
        plt.semilogy(incremental_indices, cumulative_worst_case, 
                     label=f'{label_prefix} - Worst-Case', 
                     linestyle=line_styles['worst_case'],
                     marker=marker, markersize=2, markevery=markevery,
                     linewidth=2, alpha=0.9, color=color_dark)
    
    plt.xlabel('Number of Incremental Data Points', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Gradient Residual Norm (log scale)', fontsize=14, fontweight='bold')
    plt.title(f'Gradient Norm Tracking Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=10, loc='best', framealpha=0.95, ncol=2)
    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'gradient_norm_comparison_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


def save_results_to_csv(all_results, dataset_name, output_dir):
    """
    Save gradient norm results to CSV file for multiple methods.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'method': str, 'results': dict, 'label': str}
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results for each method
    for result_dict in all_results:
        method_name = result_dict['method']
        results = result_dict['results']
        label = result_dict.get('label', method_name).replace(' ', '_').lower()
        
        grad_norm_approx = results['grad_norm_approx'].cpu().numpy()
        worst_case_bound = results['worst_case_bound_approx'].cpu().numpy()
        true_grad_norm_per_iter = results['true_grad_norm_per_iter'].cpu().numpy()
        times = results['times'].cpu().numpy()
        
        csv_path = os.path.join(output_dir, f'gradient_norm_{dataset_name.lower().replace(" ", "_")}_{label}.csv')
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['iteration', 'incremental_data_count', 'true_grad_norm',
                         'data_dependent_bound', 'worst_case_bound', 
                         'cumulative_true', 'cumulative_data_dependent', 
                         'cumulative_worst_case', 'time_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            cumulative_true = 0
            cumulative_data_dependent = 0
            cumulative_worst_case = 0
            
            for i in range(len(grad_norm_approx)):
                cumulative_true += true_grad_norm_per_iter[i]
                cumulative_data_dependent += grad_norm_approx[i]
                cumulative_worst_case += worst_case_bound[i]
                
                writer.writerow({
                    'iteration': i + 1,
                    'incremental_data_count': i + 1,
                    'true_grad_norm': true_grad_norm_per_iter[i],
                    'data_dependent_bound': grad_norm_approx[i],
                    'worst_case_bound': worst_case_bound[i],
                    'cumulative_true': cumulative_true,
                    'cumulative_data_dependent': cumulative_data_dependent,
                    'cumulative_worst_case': cumulative_worst_case,
                    'time_seconds': times[i]
                })
        
        print(f"  {method_name} results saved to: {csv_path}")
    
    # Write combined summary
    summary_path = os.path.join(output_dir, f'gradient_norm_{dataset_name.lower().replace(" ", "_")}_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'total_time_seconds', 'test_accuracy', 'num_samples',
                     'final_cumulative_true', 'final_cumulative_data_dependent', 
                     'final_cumulative_worst_case']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result_dict in all_results:
            method_name = result_dict['label']
            results = result_dict['results']
            writer.writerow({
                'method': method_name,
                'total_time_seconds': results['total_time'],
                'test_accuracy': results['test_accuracy'],
                'num_samples': results['num_samples'],
                'final_cumulative_true': results['sum_true_grad_norm'],
                'final_cumulative_data_dependent': results['sum_grad_norm'],
                'final_cumulative_worst_case': results['sum_worst_case_bound']
            })
    
    print(f"Combined summary saved to: {summary_path}")


def run_experiment(dataset='ct_slice', data_dir='./data'):
    """
    Run gradient norm tracking experiment on specified dataset.
    
    Args:
        dataset: Dataset name ('ct_slice', 'dynamic_share', or 'mnist')
        data_dir: Data directory path
    """
    print("="*80)
    print(f"GRADIENT NORM TRACKING EXPERIMENT - {dataset.upper()}")
    print("="*80)
    
    # Select dataset configuration (using gradient norm specific configs)
    if dataset == 'ct_slice':
        dataset_config = GRADIENT_NORM_DATASET_CONFIG
        load_data_fn = load_ctslice_for_experiments
        dataset_name = 'CT Slice'
    elif dataset == 'dynamic_share':
        dataset_config = GRADIENT_NORM_DATASET_CONFIG_DYNAMIC
        load_data_fn = load_dynamic_share_for_experiments
        dataset_name = 'Dynamic Share'
    elif dataset == 'mnist':
        dataset_config = GRADIENT_NORM_DATASET_CONFIG_MNIST
        load_data_fn = load_mnist_for_experiments
        dataset_name = 'MNIST'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load data
    print(f"\nLoading {dataset_name} dataset...")
    # For MNIST, we need to pass max_samples and num_base parameters
    if dataset == 'mnist':
        max_samples = dataset_config['num_training'] + dataset_config['num_removes']
        X_train, X_test, y_train, y_test = load_data_fn(
            data_dir=data_dir, 
            max_samples=max_samples,
            num_base=dataset_config['num_training']
        )
    else:
        X_train, X_test, y_train, y_test = load_data_fn(data_dir=data_dir)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Prepare arguments (using gradient norm specific args)
    args = {**dataset_config, **GRADIENT_NORM_ARGS}
    
    print(f"\nExperiment configuration:")
    print(f"  Base training samples: {args['num_training']}")
    print(f"  Incremental samples: {args['num_removes']}")
    print(f"  Lambda (regularization): {args['lam']}")
    print(f"  Std (perturbation): {args['std']}")
    print(f"  Lipschitz constant: {args['lip_constant']}")
    print(f"  Gradient norm constant: {args['gradnorm_constant']}")
    print(f"  Uniform sampling probability: {GRADIENT_NORM_UNIFORM_PROB}")
    beta, lambda_reg = GRADIENT_NORM_LEVERAGE_PARAMS
    print(f"  Leverage sampling params: β={beta}, λ={lambda_reg}")
    
    # Collect results from different methods
    all_results = []
    
    # 1. Run Incremental method (full updates)
    print("\n" + "="*80)
    print("METHOD 1: INCREMENTAL (Full Updates)")
    print("="*80)
    incremental_results = run_incremental_with_grad_norm(
        X_train.clone(), y_train.clone(), X_test, y_test, args
    )
    all_results.append({
        'method': 'incremental',
        'results': incremental_results,
        'label': 'Incremental'
    })
    
    # 2. Run Uniform sampling method
    print("\n" + "="*80)
    print(f"METHOD 2: UNIFORM SAMPLING (p={GRADIENT_NORM_UNIFORM_PROB})")
    print("="*80)
    uniform_results = run_uniform_with_grad_norm(
        X_train.clone(), y_train.clone(), X_test, y_test, 
        args, GRADIENT_NORM_UNIFORM_PROB
    )
    num_uniform_samples = uniform_results['num_samples']
    all_results.append({
        'method': 'uniform',
        'results': uniform_results,
        'label': f'Uniform ({num_uniform_samples})'
    })
    
    # 3. Run Streaming (leverage score sampling) method
    beta, lambda_reg = GRADIENT_NORM_LEVERAGE_PARAMS
    print("\n" + "="*80)
    print(f"METHOD 3: STREAMING (Leverage Score Sampling, β={beta}, λ={lambda_reg})")
    print("="*80)
    streaming_results = run_streaming_with_grad_norm(
        X_train.clone(), y_train.clone(), X_test, y_test, 
        args, beta, lambda_reg
    )
    num_streaming_samples = streaming_results['num_samples']
    all_results.append({
        'method': 'streaming',
        'results': streaming_results,
        'label': f'Streaming ({num_streaming_samples})'
    })
    
    # Create output directories
    csv_output_dir = os.path.join('results', 'csv', 'gradient_norm')
    plot_output_dir = os.path.join('results', 'plots', 'gradient_norm')
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results_to_csv(all_results, dataset_name, csv_output_dir)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOT")
    print("="*80)
    plot_gradient_norm_results(all_results, dataset_name, plot_output_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults summary:")
    for result_dict in all_results:
        results = result_dict['results']
        label = result_dict['label']
        print(f"\n{label}:")
        print(f"  Test accuracy: {results['test_accuracy']:.4f}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Samples used: {results['num_samples']}")
        print(f"  Cumulative true grad norm: {results['sum_true_grad_norm']:.4f}")
        print(f"  Cumulative data-dependent bound: {results['sum_grad_norm']:.4f}")
        print(f"  Cumulative worst-case bound: {results['sum_worst_case_bound']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run gradient norm tracking experiment')
    parser.add_argument('--dataset', type=str, default='ct_slice',
                       choices=['ct_slice', 'dynamic_share', 'mnist'],
                       help='Dataset to use (default: ct_slice)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    
    args = parser.parse_args()
    
    run_experiment(dataset=args.dataset, data_dir=args.data_dir)


if __name__ == '__main__':
    main()

