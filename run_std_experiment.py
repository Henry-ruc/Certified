# STD Parameter Experiment
# Explores the impact of std (perturbation) parameter during pretraining on test accuracy
# Runs Streaming method on MNIST with varying std values
# 
# Output:
#   - CSV files: results/csv/std_experiment/
#   - Plots: results/plots/std_experiment/

import sys
import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import math

# Import experiment utilities
from data_loader import load_mnist_for_experiments
from core_functions import (
    lr_loss, lr_grad, lr_hessian_inv, lr_hessian_inv_approximate, lr_optimize,
    spectral_norm, compute_online_leverage_score_incremental, update_XTX_inverse,
    compute_sampling_probability, device
)
from configs import GRADIENT_NORM_DATASET_CONFIG_MNIST, GRADIENT_NORM_ARGS


def run_streaming_with_accuracy_tracking(X_train, y_train, X_test, y_test, args, beta, lambda_reg):
    """
    Run streaming learning with test accuracy tracking at each incremental step.
    
    Args:
        X_train: Training data (base + incremental data)
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary with arguments (lam, std, num_training, num_removes, etc.)
        beta: Beta parameter for sampling probability
        lambda_reg: Lambda regularization parameter for leverage score
    
    Returns:
        dict: {
            'total_time': Total computation time,
            'num_samples': Total samples used,
            'num_updates': Number of actual model updates,
            'accuracy_per_iter': Array of test accuracy after each incremental data point,
            'times': Array of per-iteration times
        }
    """
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    # Calculate epsilon based on lambda_reg (delta / lambda_reg)
    delta = 0.01
    epsilon = delta / lambda_reg
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Move to device
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    # Train initial model on base data with specified std
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Evaluate initial accuracy (before any incremental data)
    pred = X_test.mv(w_approx)
    initial_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    # Initialize for incremental updates
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    
    # Track metrics
    times = torch.zeros(num_removes).float()
    accuracy_per_iter = torch.zeros(num_removes + 1).float()  # +1 for initial accuracy
    accuracy_per_iter[0] = initial_accuracy
    update_count = 0
    sum_grad_norm = 0.0
    
    start_all = time.time()
    
    for i in range(num_removes):
        start = time.time()
        
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
        
        # Original incremental data (all data seen so far)
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
        sampled = torch.rand(1).item() < sample_prob
        
        if sampled:
            # Increment update counter
            update_count += 1
            
            # Add sampled point (with rescaling)
            weighted_x = X_train[new_point_idx] / math.sqrt(sample_prob)
            X_inc_sample = torch.cat([X_inc_sample, weighted_x.unsqueeze(0)])
            y_inc_sample = torch.cat([y_inc_sample, y_train[new_point_idx].unsqueeze(0)])
            
            # Calculate Hessian inverse: combine base training set with sampled streaming data
            X_for_hessian = torch.cat([X_train[:num_training], X_inc_sample])
            y_for_hessian = torch.cat([y_train[:num_training], y_inc_sample])
            H_inv = lr_hessian_inv_approximate(w_approx, X_for_hessian, X_inc, y_for_hessian, lam)
            
            # Update Gram_sample_inverse with weighted data point
            Gram_sample_inverse = update_XTX_inverse(weighted_x, Gram_sample_inverse, lambda_reg)
            
            # Newton update
            delta_newton = H_inv.mv(Delta)
            w_approx -= delta_newton
            
            # Reset Delta
            Delta = torch.zeros_like(Delta)


        # Evaluate test accuracy after this incremental step
        pred = X_test.mv(w_approx)
        accuracy_per_iter[i + 1] = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
        
        # Compute true gradient norm using autograd on all incremental data
        w_tensor = torch.nn.Parameter(w_approx.clone())
        loss = lr_loss(w_tensor, X_inc, y_inc, lam)
        if b is not None:
            loss += b.dot(w_tensor) / X_inc.size(0)
        loss.backward()
        sum_grad_norm += w_tensor.grad.norm()

        # If sum_grad_norm exceeds threshold, retrain model on all data seen so far
        if sum_grad_norm > 100:
            print(f"    Gradient norm sum {sum_grad_norm:.2f} exceeded threshold 100, retraining model...")
            # Train model from scratch on all data seen so far (X_inc)
            w_approx = lr_optimize(X_inc, y_inc, lam, b=b, num_steps=num_steps, verbose=False)
            # Reset gradient norm sum after retraining
            sum_grad_norm = 0.0
        
        times[i] = time.time() - start
        
        if (i + 1) % 1000 == 0:
            print(f"    Completed {i+1}/{num_removes} data arrivals, {update_count} updates, accuracy: {accuracy_per_iter[i+1]:.4f}")
    
    total_time = time.time() - start_all
    
    # Calculate number of samples (base + sampled streaming)
    X_full_hessian = torch.cat([X_train[:num_training], X_inc_sample])
    num_samples = len(X_full_hessian)
    
    return {
        'total_time': total_time,
        'num_samples': num_samples,
        'num_updates': update_count,
        'accuracy_per_iter': accuracy_per_iter,
        'times': times
    }


def plot_std_experiment_results(all_results, dataset_name, output_dir):
    """
    Plot test accuracy vs incremental data size for different std values.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'std': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Color scheme for different std values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(all_results)))
    
    # Plot each std configuration
    for idx, result_dict in enumerate(all_results):
        std_val = result_dict['std']
        results = result_dict['results']
        
        # Extract accuracy data
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        num_iterations = len(accuracy_per_iter)
        
        # Create incremental data indices (starting from 0)
        incremental_indices = np.arange(num_iterations)
        
        # Plot line
        plt.plot(incremental_indices, accuracy_per_iter,
                label=f'std={std_val}',
                color=colors[idx],
                linewidth=2.5,
                marker='o',
                markersize=4,
                markevery=max(1, num_iterations // 20),
                alpha=0.85)
    
    plt.xlabel('Number of Incremental Data Points', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title(f'Impact of STD Parameter on Test Accuracy - {dataset_name}\n(Streaming Method, β={all_results[0]["beta"]}, λ={all_results[0]["lambda_reg"]})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'std_experiment_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def save_results_to_csv(all_results, dataset_name, output_dir):
    """
    Save STD experiment results to CSV file.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'std': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results for each std value
    for result_dict in all_results:
        std_val = result_dict['std']
        results = result_dict['results']
        
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        times = results['times'].cpu().numpy()
        
        csv_path = os.path.join(output_dir, f'std_experiment_{dataset_name.lower().replace(" ", "_")}_std{std_val}.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['incremental_data_count', 'test_accuracy', 'time_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write initial accuracy (before any incremental data)
            writer.writerow({
                'incremental_data_count': 0,
                'test_accuracy': accuracy_per_iter[0],
                'time_seconds': 0.0
            })
            
            # Write accuracy after each incremental data point
            for i in range(len(times)):
                writer.writerow({
                    'incremental_data_count': i + 1,
                    'test_accuracy': accuracy_per_iter[i + 1],
                    'time_seconds': times[i]
                })
        
        print(f"  std={std_val} results saved to: {csv_path}")
    
    # Write combined summary
    summary_path = os.path.join(output_dir, f'std_experiment_{dataset_name.lower().replace(" ", "_")}_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['std', 'total_time_seconds', 'final_accuracy', 'num_samples', 'num_updates']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result_dict in all_results:
            std_val = result_dict['std']
            results = result_dict['results']
            accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
            
            writer.writerow({
                'std': std_val,
                'total_time_seconds': results['total_time'],
                'final_accuracy': accuracy_per_iter[-1],
                'num_samples': results['num_samples'],
                'num_updates': results['num_updates']
            })
    
    print(f"Combined summary saved to: {summary_path}")


def run_experiment(std_values=None, beta=1.02, lambda_reg=0.014, data_dir='./data'):
    """
    Run STD parameter experiment on MNIST dataset.
    
    Args:
        std_values: List of std values to test (default: [5, 10, 20, 40, 80])
        beta: Beta parameter for streaming method
        lambda_reg: Lambda regularization parameter for leverage score
        data_dir: Data directory path
    """
    if std_values is None:
        std_values = [5, 10, 20, 40, 80]
    
    print("="*80)
    print("STD PARAMETER EXPERIMENT - MNIST")
    print("="*80)
    print(f"Testing std values: {std_values}")
    print(f"Fixed parameters: β={beta}, λ={lambda_reg}")
    
    # Load MNIST data
    dataset_config = GRADIENT_NORM_DATASET_CONFIG_MNIST
    dataset_name = 'MNIST'
    
    print(f"\nLoading {dataset_name} dataset...")
    max_samples = dataset_config['num_training'] + dataset_config['num_removes']
    X_train, X_test, y_train, y_test = load_mnist_for_experiments(
        data_dir=data_dir,
        max_samples=max_samples,
        num_base=dataset_config['num_training']
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Prepare base arguments
    base_args = {**dataset_config, **GRADIENT_NORM_ARGS}
    
    print(f"\nExperiment configuration:")
    print(f"  Base training samples: {base_args['num_training']}")
    print(f"  Incremental samples: {base_args['num_removes']}")
    print(f"  Lambda (regularization): {base_args['lam']}")
    print(f"  Streaming params: β={beta}, λ={lambda_reg}")
    
    # Collect results for different std values
    all_results = []
    
    for std_val in std_values:
        print("\n" + "="*80)
        print(f"Running with std={std_val}")
        print("="*80)
        
        # Update args with current std value
        args = base_args.copy()
        args['std'] = std_val
        
        # Run streaming experiment with accuracy tracking
        results = run_streaming_with_accuracy_tracking(
            X_train.clone(), y_train.clone(), X_test, y_test,
            args, beta, lambda_reg
        )
        
        all_results.append({
            'std': std_val,
            'beta': beta,
            'lambda_reg': lambda_reg,
            'results': results
        })
        
        print(f"  Completed: Time={results['total_time']:.2f}s, "
              f"Final Accuracy={results['accuracy_per_iter'][-1]:.4f}, "
              f"Samples={results['num_samples']}, "
              f"Updates={results['num_updates']}")
    
    # Create output directories
    csv_output_dir = os.path.join('results', 'csv', 'std_experiment')
    plot_output_dir = os.path.join('results', 'plots', 'std_experiment')
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_results_to_csv(all_results, dataset_name, csv_output_dir)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOT")
    print("="*80)
    plot_std_experiment_results(all_results, dataset_name, plot_output_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults summary:")
    for result_dict in all_results:
        std_val = result_dict['std']
        results = result_dict['results']
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        print(f"\nstd={std_val}:")
        print(f"  Initial accuracy: {accuracy_per_iter[0]:.4f}")
        print(f"  Final accuracy: {accuracy_per_iter[-1]:.4f}")
        print(f"  Accuracy change: {accuracy_per_iter[-1] - accuracy_per_iter[0]:+.4f}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Samples used: {results['num_samples']}")
        print(f"  Updates: {results['num_updates']}")


def run_lam_experiment(lam_values=None, beta=1.02, lambda_reg=0.014, data_dir='./data'):
    """
    Run Lambda (regularization) parameter experiment on MNIST dataset.
    
    Args:
        lam_values: List of lam values to test (default: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        beta: Beta parameter for streaming method
        lambda_reg: Lambda regularization parameter for leverage score
        data_dir: Data directory path
    """
    if lam_values is None:
        lam_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    
    print("="*80)
    print("LAMBDA (REGULARIZATION) PARAMETER EXPERIMENT - MNIST")
    print("="*80)
    print(f"Testing lam values: {lam_values}")
    print(f"Fixed parameters: β={beta}, λ={lambda_reg}")
    
    # Load MNIST data
    dataset_config = GRADIENT_NORM_DATASET_CONFIG_MNIST
    dataset_name = 'MNIST'
    
    print(f"\nLoading {dataset_name} dataset...")
    max_samples = dataset_config['num_training'] + dataset_config['num_removes']
    X_train, X_test, y_train, y_test = load_mnist_for_experiments(
        data_dir=data_dir,
        max_samples=max_samples,
        num_base=dataset_config['num_training']
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Prepare base arguments
    base_args = {**dataset_config, **GRADIENT_NORM_ARGS}
    # Fix std to 10 for lam experiment
    base_args['std'] = 10
    
    print(f"\nExperiment configuration:")
    print(f"  Base training samples: {base_args['num_training']}")
    print(f"  Incremental samples: {base_args['num_removes']}")
    print(f"  STD (perturbation): {base_args['std']}")
    print(f"  Streaming params: β={beta}, λ={lambda_reg}")
    
    # Collect results for different lam values
    all_results = []
    
    for lam_val in lam_values:
        print("\n" + "="*80)
        print(f"Running with lam={lam_val}")
        print("="*80)
        
        # Update args with current lam value
        args = base_args.copy()
        args['lam'] = lam_val
        
        # Run streaming experiment with accuracy tracking
        results = run_streaming_with_accuracy_tracking(
            X_train.clone(), y_train.clone(), X_test, y_test,
            args, beta, lambda_reg
        )
        
        all_results.append({
            'lam': lam_val,
            'beta': beta,
            'lambda_reg': lambda_reg,
            'results': results
        })
        
        print(f"  Completed: Time={results['total_time']:.2f}s, "
              f"Final Accuracy={results['accuracy_per_iter'][-1]:.4f}, "
              f"Samples={results['num_samples']}, "
              f"Updates={results['num_updates']}")
    
    # Create output directories
    csv_output_dir = os.path.join('results', 'csv', 'lam_experiment')
    plot_output_dir = os.path.join('results', 'plots', 'lam_experiment')
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_lam_results_to_csv(all_results, dataset_name, csv_output_dir)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOT")
    print("="*80)
    plot_lam_experiment_results(all_results, dataset_name, plot_output_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults summary:")
    for result_dict in all_results:
        lam_val = result_dict['lam']
        results = result_dict['results']
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        print(f"\nlam={lam_val:.0e}:")
        print(f"  Initial accuracy: {accuracy_per_iter[0]:.4f}")
        print(f"  Final accuracy: {accuracy_per_iter[-1]:.4f}")
        print(f"  Accuracy change: {accuracy_per_iter[-1] - accuracy_per_iter[0]:+.4f}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Samples used: {results['num_samples']}")
        print(f"  Updates: {results['num_updates']}")


def plot_lam_experiment_results(all_results, dataset_name, output_dir):
    """
    Plot test accuracy vs incremental data size for different lam values.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'lam': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Color scheme for different lam values
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(all_results)))
    
    # Plot each lam configuration
    for idx, result_dict in enumerate(all_results):
        lam_val = result_dict['lam']
        results = result_dict['results']
        
        # Extract accuracy data
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        num_iterations = len(accuracy_per_iter)
        
        # Create incremental data indices (starting from 0)
        incremental_indices = np.arange(num_iterations)
        
        # Plot line
        plt.plot(incremental_indices, accuracy_per_iter,
                label=f'lam={lam_val:.0e}',
                color=colors[idx],
                linewidth=2.5,
                marker='s',
                markersize=4,
                markevery=max(1, num_iterations // 20),
                alpha=0.85)
    
    plt.xlabel('Number of Incremental Data Points', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title(f'Impact of Lambda (Regularization) on Test Accuracy - {dataset_name}\n(Streaming Method, β={all_results[0]["beta"]}, λ={all_results[0]["lambda_reg"]})', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'lam_experiment_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def save_lam_results_to_csv(all_results, dataset_name, output_dir):
    """
    Save Lambda experiment results to CSV file.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'lam': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results for each lam value
    for result_dict in all_results:
        lam_val = result_dict['lam']
        results = result_dict['results']
        
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        times = results['times'].cpu().numpy()
        
        csv_path = os.path.join(output_dir, f'lam_experiment_{dataset_name.lower().replace(" ", "_")}_lam{lam_val:.0e}.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['incremental_data_count', 'test_accuracy', 'time_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write initial accuracy (before any incremental data)
            writer.writerow({
                'incremental_data_count': 0,
                'test_accuracy': accuracy_per_iter[0],
                'time_seconds': 0.0
            })
            
            # Write accuracy after each incremental data point
            for i in range(len(times)):
                writer.writerow({
                    'incremental_data_count': i + 1,
                    'test_accuracy': accuracy_per_iter[i + 1],
                    'time_seconds': times[i]
                })
        
        print(f"  lam={lam_val:.0e} results saved to: {csv_path}")
    
    # Write combined summary
    summary_path = os.path.join(output_dir, f'lam_experiment_{dataset_name.lower().replace(" ", "_")}_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['lam', 'total_time_seconds', 'final_accuracy', 'num_samples', 'num_updates']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result_dict in all_results:
            lam_val = result_dict['lam']
            results = result_dict['results']
            accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
            
            writer.writerow({
                'lam': lam_val,
                'total_time_seconds': results['total_time'],
                'final_accuracy': accuracy_per_iter[-1],
                'num_samples': results['num_samples'],
                'num_updates': results['num_updates']
            })
    
    print(f"Combined summary saved to: {summary_path}")


def run_combined_experiment(std_values=None, lam_values=None, beta=1.02, lambda_reg=0.014, data_dir='./data'):
    """
    Run combined STD and Lambda experiment on MNIST dataset.
    Tests all combinations of std and lam values and plots them in one figure.
    
    Args:
        std_values: List of std values to test
        lam_values: List of lam values to test (for each std)
        beta: Beta parameter for streaming method
        lambda_reg: Lambda regularization parameter for leverage score
        data_dir: Data directory path
    """
    if std_values is None:
        std_values = [10, 20, 40]
    if lam_values is None:
        lam_values = [1e-4, 1e-3, 1e-2]
    
    print("="*80)
    print("COMBINED STD & LAMBDA PARAMETER EXPERIMENT - MNIST")
    print("="*80)
    print(f"Testing std values: {std_values}")
    print(f"Testing lam values: {lam_values}")
    print(f"Total combinations: {len(std_values)} × {len(lam_values)} = {len(std_values) * len(lam_values)}")
    print(f"Fixed parameters: β={beta}, λ={lambda_reg}")
    
    # Load MNIST data
    dataset_config = GRADIENT_NORM_DATASET_CONFIG_MNIST
    dataset_name = 'MNIST'
    
    print(f"\nLoading {dataset_name} dataset...")
    max_samples = dataset_config['num_training'] + dataset_config['num_removes']
    X_train, X_test, y_train, y_test = load_mnist_for_experiments(
        data_dir=data_dir,
        max_samples=max_samples,
        num_base=dataset_config['num_training']
    )
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Prepare base arguments
    base_args = {**dataset_config, **GRADIENT_NORM_ARGS}
    
    print(f"\nExperiment configuration:")
    print(f"  Base training samples: {base_args['num_training']}")
    print(f"  Incremental samples: {base_args['num_removes']}")
    print(f"  Streaming params: β={beta}, λ={lambda_reg}")
    
    # Collect results for all combinations
    all_results = []
    
    for std_val in std_values:
        for lam_val in lam_values:
            print("\n" + "="*80)
            print(f"Running with std={std_val}, lam={lam_val:.0e}")
            print("="*80)
            
            # Update args with current std and lam values
            args = base_args.copy()
            args['std'] = std_val
            args['lam'] = lam_val
            
            # Run streaming experiment with accuracy tracking
            results = run_streaming_with_accuracy_tracking(
                X_train.clone(), y_train.clone(), X_test, y_test,
                args, beta, lambda_reg
            )
            
            all_results.append({
                'std': std_val,
                'lam': lam_val,
                'beta': beta,
                'lambda_reg': lambda_reg,
                'results': results
            })
            
            print(f"  Completed: Time={results['total_time']:.2f}s, "
                  f"Final Accuracy={results['accuracy_per_iter'][-1]:.4f}, "
                  f"Samples={results['num_samples']}, "
                  f"Updates={results['num_updates']}")
    
    # Create output directories
    csv_output_dir = os.path.join('results', 'csv', 'combined_experiment')
    plot_output_dir = os.path.join('results', 'plots', 'combined_experiment')
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    save_combined_results_to_csv(all_results, dataset_name, csv_output_dir)
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOT")
    print("="*80)
    plot_combined_experiment_results(all_results, dataset_name, plot_output_dir, std_values, lam_values)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"\nResults summary:")
    for result_dict in all_results:
        std_val = result_dict['std']
        lam_val = result_dict['lam']
        results = result_dict['results']
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        print(f"\nstd={std_val}, lam={lam_val:.0e}:")
        print(f"  Initial accuracy: {accuracy_per_iter[0]:.4f}")
        print(f"  Final accuracy: {accuracy_per_iter[-1]:.4f}")
        print(f"  Accuracy change: {accuracy_per_iter[-1] - accuracy_per_iter[0]:+.4f}")
        print(f"  Total time: {results['total_time']:.2f}s")
        print(f"  Samples used: {results['num_samples']}")


def plot_combined_experiment_results(all_results, dataset_name, output_dir, std_values, lam_values):
    """
    Plot test accuracy vs incremental data size for all (std, lam) combinations in one figure.
    Each std value uses a different color, and each lam value uses a different line style.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'std': float, 'lam': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot
        std_values: List of std values (for ordering)
        lam_values: List of lam values (for ordering)
    """
    plt.figure(figsize=(14, 9))
    
    # Color schemes for different std values - use different color families
    # Each std gets a base color from a different colormap
    color_maps = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples', 'Greys']
    
    # Get color for each std value from different colormaps
    std_colors = []
    for idx in range(len(std_values)):
        cmap = plt.cm.get_cmap(color_maps[idx % len(color_maps)])
        # Use darker colors from the colormap (0.6-0.9 range)
        std_colors.append(cmap(0.7))
    
    # Line styles for different lam values - make them easily distinguishable
    line_styles = ['-', '--', '-.', ':']
    # Extend line styles if needed
    while len(line_styles) < len(lam_values):
        line_styles.extend(['-', '--', '-.', ':'])
    
    # Markers for different lam values
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    while len(markers) < len(lam_values):
        markers.extend(['o', 's', '^', 'D', 'v', '<', '>', 'p'])
    
    # Line widths vary by lam (heavier lines for larger lam values)
    line_widths = np.linspace(2.0, 3.5, len(lam_values))
    
    # Plot each combination
    for result_dict in all_results:
        std_val = result_dict['std']
        lam_val = result_dict['lam']
        results = result_dict['results']
        
        # Find indices for color and line style
        std_idx = std_values.index(std_val)
        lam_idx = lam_values.index(lam_val)
        
        # Extract accuracy data
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        num_iterations = len(accuracy_per_iter)
        
        # Create incremental data indices (starting from 0)
        incremental_indices = np.arange(num_iterations)
        
        # Plot line - same color for same std, different line style for different lam
        plt.plot(incremental_indices, accuracy_per_iter,
                label=f'std={std_val}, lam={lam_val:.0e}',
                color=std_colors[std_idx],
                linestyle=line_styles[lam_idx],
                linewidth=line_widths[lam_idx],
                marker=markers[lam_idx],
                markersize=4,
                markevery=max(1, num_iterations // 20),
                alpha=0.9)
    
    plt.xlabel('Number of Incremental Data Points', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    plt.title(f'Impact of STD and Lambda on Test Accuracy - {dataset_name}\n(Streaming Method, β={all_results[0]["beta"]}, λ={all_results[0]["lambda_reg"]})', 
              fontsize=15, fontweight='bold')
    
    # Create legend with multiple columns for better readability
    # Group by std value in legend
    num_cols = len(lam_values)
    plt.legend(fontsize=10, loc='best', framealpha=0.95, ncol=num_cols)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'combined_experiment_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    plt.close()


def save_combined_results_to_csv(all_results, dataset_name, output_dir):
    """
    Save combined experiment results to CSV file.
    
    Args:
        all_results: List of dictionaries, each containing:
                    {'std': float, 'lam': float, 'results': dict}
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual results for each (std, lam) combination
    for result_dict in all_results:
        std_val = result_dict['std']
        lam_val = result_dict['lam']
        results = result_dict['results']
        
        accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
        times = results['times'].cpu().numpy()
        
        csv_path = os.path.join(output_dir, 
                               f'combined_experiment_{dataset_name.lower().replace(" ", "_")}_std{std_val}_lam{lam_val:.0e}.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['incremental_data_count', 'test_accuracy', 'time_seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Write initial accuracy (before any incremental data)
            writer.writerow({
                'incremental_data_count': 0,
                'test_accuracy': accuracy_per_iter[0],
                'time_seconds': 0.0
            })
            
            # Write accuracy after each incremental data point
            for i in range(len(times)):
                writer.writerow({
                    'incremental_data_count': i + 1,
                    'test_accuracy': accuracy_per_iter[i + 1],
                    'time_seconds': times[i]
                })
        
        print(f"  std={std_val}, lam={lam_val:.0e} results saved to: {csv_path}")
    
    # Write combined summary
    summary_path = os.path.join(output_dir, f'combined_experiment_{dataset_name.lower().replace(" ", "_")}_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['std', 'lam', 'total_time_seconds', 'initial_accuracy', 'final_accuracy', 
                     'accuracy_change', 'num_samples', 'num_updates']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result_dict in all_results:
            std_val = result_dict['std']
            lam_val = result_dict['lam']
            results = result_dict['results']
            accuracy_per_iter = results['accuracy_per_iter'].cpu().numpy()
            
            writer.writerow({
                'std': std_val,
                'lam': lam_val,
                'total_time_seconds': results['total_time'],
                'initial_accuracy': accuracy_per_iter[0],
                'final_accuracy': accuracy_per_iter[-1],
                'accuracy_change': accuracy_per_iter[-1] - accuracy_per_iter[0],
                'num_samples': results['num_samples'],
                'num_updates': results['num_updates']
            })
    
    print(f"Combined summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Run pretraining parameter experiments (STD, Lambda, or Combined)')
    parser.add_argument('--experiment-type', type=str, default='std',
                       choices=['std', 'lam', 'combined', 'all'],
                       help='Type of experiment to run: std, lam, combined, or all (default: std)')
    parser.add_argument('--std-values', type=float, nargs='+', default=[0, 5, 10, 20, 40],
                       help='List of std values to test (default: [5, 10, 20, 40, 80])')
    parser.add_argument('--lam-values', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                       help='List of lam values to test (default: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])')
    parser.add_argument('--beta', type=float, default=1.02,
                       help='Beta parameter for streaming (default: 1.02)')
    parser.add_argument('--lambda-reg', type=float, default=0.014,
                       help='Lambda regularization for leverage score (default: 0.014)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    
    args = parser.parse_args()
    
    if args.experiment_type == 'std' or args.experiment_type == 'all':
        run_experiment(
            std_values=args.std_values,
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            data_dir=args.data_dir
        )
    
    if args.experiment_type == 'lam' or args.experiment_type == 'all':
        if args.experiment_type == 'all':
            print("\n\n")  # Add some spacing between experiments
        run_lam_experiment(
            lam_values=args.lam_values,
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            data_dir=args.data_dir
        )
    
    if args.experiment_type == 'combined' or args.experiment_type == 'all':
        if args.experiment_type == 'all':
            print("\n\n")  # Add some spacing between experiments
        run_combined_experiment(
            std_values=args.std_values,
            lam_values=args.lam_values,
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            data_dir=args.data_dir
        )


if __name__ == '__main__':
    main()

