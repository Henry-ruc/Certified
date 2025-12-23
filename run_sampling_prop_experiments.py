# Sampling Property Analysis Experiment
# Analyzes and visualizes data points with highest leverage scores during streaming
# 
# Output:
#   - CSV files: results/csv/sampling_analysis/
#   - Images: results/plots/sampling_analysis/

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


def run_streaming_with_leverage_tracking(X_train, y_train, X_test, y_test, args, beta, lambda_reg, top_k=5):
    """
    Run streaming learning with leverage score tracking to identify top-k highest leverage score samples.
    
    Args:
        X_train: Training data (base + incremental data)
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary with arguments (lam, std, num_training, num_removes, etc.)
        beta: Beta parameter for sampling probability
        lambda_reg: Lambda regularization parameter for leverage score
        top_k: Number of top leverage score samples to track
    
    Returns:
        dict: {
            'total_time': Total computation time,
            'test_accuracy': Test set accuracy,
            'num_samples': Total samples used,
            'num_updates': Number of actual model updates,
            'top_k_indices': Indices of top-k highest leverage score data points,
            'top_k_leverage_scores': Leverage scores of top-k data points,
            'top_k_sample_probs': Sampling probabilities of top-k data points,
            'top_k_images': Image data of top-k data points,
            'top_k_labels': Labels of top-k data points,
            'all_leverage_scores': All leverage scores for analysis
        }
    """
    print(f"\n{'='*80}")
    print(f"Running STREAMING with Leverage Score Tracking (β={beta}, λ={lambda_reg})")
    print(f"{'='*80}")
    
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    # Calculate epsilon based on lambda_reg (delta / lambda_reg)
    delta = 0.01
    epsilon = delta / lambda_reg
    
    print(f"Parameters: lam={lam}, std={std}, beta={beta}, lambda_reg={lambda_reg}, epsilon={epsilon}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Move to device
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    
    X_train = X_train.float().to(device)
    y_train = y_train.float().to(device)
    X_test = X_test.float().to(device)
    y_test = y_test.to(device)
    
    # Train initial model on base data
    print(f"Training initial model on {num_training} base samples...")
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Initialize for incremental updates
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    
    # Track metrics
    update_count = 0
    all_leverage_scores = []
    all_sample_probs = []
    all_indices = []
    
    # Track top-k leverage scores
    top_k_data = {
        'indices': [],
        'leverage_scores': [],
        'sample_probs': [],
        'images': [],
        'labels': []
    }
    
    # Track bottom-k leverage scores
    bottom_k_data = {
        'indices': [],
        'leverage_scores': [],
        'sample_probs': [],
        'images': [],
        'labels': []
    }
    
    # Skip first 1000 incremental data points for top-k and bottom-k tracking
    skip_first_n = 1000
    
    print(f"\nPerforming {num_removes} data arrivals with leverage score tracking...")
    print(f"Note: Skipping first {skip_first_n} incremental data points for top-k tracking")
    start_all = time.time()
    
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
        
        # Calculate the leverage score and sampling probability
        leverage_score = compute_online_leverage_score_incremental(
            X_train[new_point_idx], Gram_sample_inverse, X_inc_sample, epsilon, lambda_reg)
        sample_prob = compute_sampling_probability(leverage_score, d, beta, epsilon)
        
        # Store leverage score and sampling probability
        all_leverage_scores.append(leverage_score)
        all_sample_probs.append(sample_prob)
        all_indices.append(new_point_idx)
        
        # Update top-k and bottom-k tracking (skip first skip_first_n incremental data points)
        if i >= skip_first_n:
            # Track top-k (highest leverage scores)
            if len(top_k_data['leverage_scores']) < top_k:
                # Still filling up top-k
                top_k_data['indices'].append(new_point_idx)
                top_k_data['leverage_scores'].append(leverage_score)
                top_k_data['sample_probs'].append(sample_prob)
                top_k_data['images'].append(X_train[new_point_idx].cpu())
                top_k_data['labels'].append(y_train[new_point_idx].cpu().item())
            else:
                # Check if current leverage score is higher than or equal to minimum in top-k
                # Using >= ensures that when leverage scores are equal, later points replace earlier ones
                min_idx = np.argmin(top_k_data['leverage_scores'])
                if leverage_score >= top_k_data['leverage_scores'][min_idx]:
                    # Replace the minimum
                    top_k_data['indices'][min_idx] = new_point_idx
                    top_k_data['leverage_scores'][min_idx] = leverage_score
                    top_k_data['sample_probs'][min_idx] = sample_prob
                    top_k_data['images'][min_idx] = X_train[new_point_idx].cpu()
                    top_k_data['labels'][min_idx] = y_train[new_point_idx].cpu().item()
            
            # Track bottom-k (lowest leverage scores)
            if len(bottom_k_data['leverage_scores']) < top_k:
                # Still filling up bottom-k
                bottom_k_data['indices'].append(new_point_idx)
                bottom_k_data['leverage_scores'].append(leverage_score)
                bottom_k_data['sample_probs'].append(sample_prob)
                bottom_k_data['images'].append(X_train[new_point_idx].cpu())
                bottom_k_data['labels'].append(y_train[new_point_idx].cpu().item())
            else:
                # Check if current leverage score is lower than or equal to maximum in bottom-k
                # Using <= ensures that when leverage scores are equal, later points replace earlier ones
                max_idx = np.argmax(bottom_k_data['leverage_scores'])
                if leverage_score <= bottom_k_data['leverage_scores'][max_idx]:
                    # Replace the maximum
                    bottom_k_data['indices'][max_idx] = new_point_idx
                    bottom_k_data['leverage_scores'][max_idx] = leverage_score
                    bottom_k_data['sample_probs'][max_idx] = sample_prob
                    bottom_k_data['images'][max_idx] = X_train[new_point_idx].cpu()
                    bottom_k_data['labels'][max_idx] = y_train[new_point_idx].cpu().item()
        
        # Sample point with probability sample_prob
        if torch.rand(1).item() < sample_prob:
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
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count} updates...")
    
    total_time = time.time() - start_all
    
    # Evaluate on test set
    pred = X_test.mv(w_approx)
    test_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    # Calculate number of samples (base + sampled streaming)
    X_full_hessian = torch.cat([X_train[:num_training], X_inc_sample])
    num_samples = len(X_full_hessian)
    
    print(f"\nStreaming Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Total samples used: {num_samples}")
    print(f"  Actual updates: {update_count}/{num_removes} ({100*update_count/num_removes:.1f}%)")
    
    # Sort top-k by leverage score (descending)
    sorted_indices_top = np.argsort(top_k_data['leverage_scores'])[::-1]
    for key in top_k_data:
        if isinstance(top_k_data[key], list):
            top_k_data[key] = [top_k_data[key][i] for i in sorted_indices_top]
    
    # Sort bottom-k by leverage score (ascending)
    sorted_indices_bottom = np.argsort(bottom_k_data['leverage_scores'])
    for key in bottom_k_data:
        if isinstance(bottom_k_data[key], list):
            bottom_k_data[key] = [bottom_k_data[key][i] for i in sorted_indices_bottom]
    
    print(f"\nTop-{top_k} Highest Leverage Score Samples (from incremental data point {skip_first_n+1} onwards):")
    for i in range(len(top_k_data['indices'])):
        print(f"  Rank {i+1}: Index={top_k_data['indices'][i]}, "
              f"Leverage Score={top_k_data['leverage_scores'][i]:.6f}, "
              f"Sample Prob={top_k_data['sample_probs'][i]:.6f}, "
              f"Label={top_k_data['labels'][i]}")
    
    print(f"\nBottom-{top_k} Lowest Leverage Score Samples (from incremental data point {skip_first_n+1} onwards):")
    for i in range(len(bottom_k_data['indices'])):
        print(f"  Rank {i+1}: Index={bottom_k_data['indices'][i]}, "
              f"Leverage Score={bottom_k_data['leverage_scores'][i]:.6f}, "
              f"Sample Prob={bottom_k_data['sample_probs'][i]:.6f}, "
              f"Label={bottom_k_data['labels'][i]}")
    
    return {
        'total_time': total_time,
        'test_accuracy': test_accuracy,
        'num_samples': num_samples,
        'num_updates': update_count,
        'skip_first_n': skip_first_n,
        'top_k_indices': top_k_data['indices'],
        'top_k_leverage_scores': top_k_data['leverage_scores'],
        'top_k_sample_probs': top_k_data['sample_probs'],
        'top_k_images': top_k_data['images'],
        'top_k_labels': top_k_data['labels'],
        'bottom_k_indices': bottom_k_data['indices'],
        'bottom_k_leverage_scores': bottom_k_data['leverage_scores'],
        'bottom_k_sample_probs': bottom_k_data['sample_probs'],
        'bottom_k_images': bottom_k_data['images'],
        'bottom_k_labels': bottom_k_data['labels'],
        'all_leverage_scores': all_leverage_scores,
        'all_sample_probs': all_sample_probs,
        'all_indices': all_indices
    }


def visualize_top_and_bottom_k_samples(results, dataset_name, output_dir):
    """
    Visualize both top-k and bottom-k samples in one figure with two rows.
    Top row: highest leverage score samples
    Bottom row: lowest leverage score samples
    
    Args:
        results: Results dictionary from run_streaming_with_leverage_tracking
        dataset_name: Name of the dataset
        output_dir: Directory to save the visualization
    """
    k = len(results['top_k_images'])
    
    # Create figure with 2 rows and k columns
    fig, axes = plt.subplots(2, k, figsize=(3*k, 7))
    
    # Top row: Top-k samples
    for i in range(k):
        img = results['top_k_images'][i].numpy()
        
        # Reshape to 28x28 for MNIST
        if len(img.shape) == 1:
            img_size = int(np.sqrt(len(img)))
            img = img.reshape(img_size, img_size)
        
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
    
    # Add row label for Top (consistent with regenerate_plots.py style)
    axes[0, 0].text(-0.15, 0.5, f'Top {k}', transform=axes[0, 0].transAxes,
                   fontsize=23, fontfamily='serif', color='#424242',
                   va='center', ha='right')
    
    # Bottom row: Bottom-k samples
    for i in range(k):
        img = results['bottom_k_images'][i].numpy()
        
        # Reshape to 28x28 for MNIST
        if len(img.shape) == 1:
            img_size = int(np.sqrt(len(img)))
            img = img.reshape(img_size, img_size)
        
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
    
    # Add row label for Bottom (consistent with regenerate_plots.py style)
    axes[1, 0].text(-0.15, 0.5, f'Bottom {k}', transform=axes[1, 0].transAxes,
                   fontsize=23, fontfamily='serif', color='#424242',
                   va='center', ha='right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'top_bottom_{k}_leverage_samples_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Top-{k} and Bottom-{k} samples visualization saved to: {plot_path}")
    
    # Save PDF
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Top-{k} and Bottom-{k} samples visualization (PDF) saved to: {pdf_path}")
    
    plt.close()


def plot_leverage_score_distribution(results, dataset_name, output_dir):
    """
    Plot the distribution of leverage scores across all incremental data points.
    
    Args:
        results: Results dictionary from run_streaming_with_leverage_tracking
        dataset_name: Name of the dataset
        output_dir: Directory to save the plot
    """
    leverage_scores = results['all_leverage_scores']
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(leverage_scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Leverage Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Leverage Score Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark top-k scores
    top_k_scores = results['top_k_leverage_scores']
    for i, score in enumerate(top_k_scores):
        plt.axvline(x=score, color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(score, plt.ylim()[1]*0.9, f'Top{i+1}', rotation=90, 
                va='top', fontsize=8, color='red')
    
    # Mark bottom-k scores
    bottom_k_scores = results['bottom_k_leverage_scores']
    for i, score in enumerate(bottom_k_scores):
        plt.axvline(x=score, color='blue', linestyle='--', linewidth=1, alpha=0.7)
        plt.text(score, plt.ylim()[1]*0.8, f'Bot{i+1}', rotation=90, 
                va='top', fontsize=8, color='blue')
    
    # Subplot 2: Leverage score over time
    plt.subplot(1, 2, 2)
    indices = results['all_indices']
    skip_first_n = results.get('skip_first_n', 0)
    plt.plot(indices, leverage_scores, alpha=0.5, linewidth=0.5, color='steelblue')
    plt.scatter(results['top_k_indices'], results['top_k_leverage_scores'], 
               color='red', s=100, zorder=5, label=f'Top-{len(top_k_scores)}', marker='*')
    plt.scatter(results['bottom_k_indices'], results['bottom_k_leverage_scores'], 
               color='blue', s=100, zorder=5, label=f'Bottom-{len(bottom_k_scores)}', marker='v')
    
    # Mark the skip boundary
    if skip_first_n > 0:
        skip_index = indices[skip_first_n] if skip_first_n < len(indices) else indices[-1]
        plt.axvline(x=skip_index, color='orange', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Skip first {skip_first_n}')
    
    plt.xlabel('Data Point Index', fontsize=12, fontweight='bold')
    plt.ylabel('Leverage Score', fontsize=12, fontweight='bold')
    plt.title('Leverage Score Over Incremental Data', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'leverage_score_distribution_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Leverage score distribution plot saved to: {plot_path}")
    
    # Save PDF
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Leverage score distribution plot (PDF) saved to: {pdf_path}")
    
    plt.close()


def save_results_to_csv(results, dataset_name, output_dir):
    """
    Save sampling analysis results to CSV file.
    
    Args:
        results: Results dictionary from run_streaming_with_leverage_tracking
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save top-k samples
    top_k_path = os.path.join(output_dir, f'top_k_samples_{dataset_name.lower().replace(" ", "_")}.csv')
    with open(top_k_path, 'w', newline='') as csvfile:
        fieldnames = ['rank', 'index', 'leverage_score', 'sample_prob', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(results['top_k_indices'])):
            writer.writerow({
                'rank': i + 1,
                'index': results['top_k_indices'][i],
                'leverage_score': results['top_k_leverage_scores'][i],
                'sample_prob': results['top_k_sample_probs'][i],
                'label': results['top_k_labels'][i]
            })
    
    print(f"Top-k samples data saved to: {top_k_path}")
    
    # Save bottom-k samples
    bottom_k_path = os.path.join(output_dir, f'bottom_k_samples_{dataset_name.lower().replace(" ", "_")}.csv')
    with open(bottom_k_path, 'w', newline='') as csvfile:
        fieldnames = ['rank', 'index', 'leverage_score', 'sample_prob', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(results['bottom_k_indices'])):
            writer.writerow({
                'rank': i + 1,
                'index': results['bottom_k_indices'][i],
                'leverage_score': results['bottom_k_leverage_scores'][i],
                'sample_prob': results['bottom_k_sample_probs'][i],
                'label': results['bottom_k_labels'][i]
            })
    
    print(f"Bottom-k samples data saved to: {bottom_k_path}")
    
    # Save all leverage scores
    all_scores_path = os.path.join(output_dir, f'all_leverage_scores_{dataset_name.lower().replace(" ", "_")}.csv')
    with open(all_scores_path, 'w', newline='') as csvfile:
        fieldnames = ['index', 'leverage_score', 'sample_prob']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i in range(len(results['all_leverage_scores'])):
            writer.writerow({
                'index': results['all_indices'][i],
                'leverage_score': results['all_leverage_scores'][i],
                'sample_prob': results['all_sample_probs'][i]
            })
    
    print(f"All leverage scores saved to: {all_scores_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, f'sampling_analysis_summary_{dataset_name.lower().replace(" ", "_")}.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['metric', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({'metric': 'total_time_seconds', 'value': results['total_time']})
        writer.writerow({'metric': 'test_accuracy', 'value': results['test_accuracy']})
        writer.writerow({'metric': 'num_samples', 'value': results['num_samples']})
        writer.writerow({'metric': 'num_updates', 'value': results['num_updates']})
        writer.writerow({'metric': 'skip_first_n_incremental_points', 'value': results.get('skip_first_n', 0)})
        writer.writerow({'metric': 'mean_leverage_score', 'value': np.mean(results['all_leverage_scores'])})
        writer.writerow({'metric': 'std_leverage_score', 'value': np.std(results['all_leverage_scores'])})
        writer.writerow({'metric': 'max_leverage_score', 'value': np.max(results['all_leverage_scores'])})
        writer.writerow({'metric': 'min_leverage_score', 'value': np.min(results['all_leverage_scores'])})
    
    print(f"Summary saved to: {summary_path}")


def run_streaming_uniform_delta_newton_comparison(X_train, y_train, X_test, y_test, args, beta, lambda_reg, uniform_prob):
    """
    Compare delta_newton norms between Streaming and Uniform methods.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary of arguments
        beta: Beta parameter for Streaming method
        lambda_reg: Lambda regularization parameter for Streaming method
        uniform_prob: Sampling probability for Uniform method
    
    Returns:
        dict: Statistics of delta_newton norms for both methods
    """
    from run_uniform import run_uniform_experiment
    from run_streaming import run_streaming_experiment
    
    print(f"\n{'='*80}")
    print("Comparing Delta Newton Norms: Streaming vs Uniform Methods")
    print(f"{'='*80}")
    
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    # Calculate epsilon
    delta = 0.01
    epsilon = delta / lambda_reg
    
    results = {'streaming': {}, 'uniform': {}}
    
    # ===== Run Streaming Method =====
    print(f"\nRunning STREAMING method (β={beta}, λ={lambda_reg})...")
    torch.manual_seed(42)
    
    X_train_base = X_train[:num_training].float().to(device)
    y_train_base = y_train[:num_training].float().to(device)
    X_train_device = X_train.float().to(device)
    y_train_device = y_train.float().to(device)
    X_test_device = X_test.float().to(device)
    y_test_device = y_test.to(device)
    
    # Train initial model
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx_stream = w.clone()
    
    # Initialize
    d = X_train_device.shape[1]
    Delta = torch.zeros(d).to(device)
    X_inc = X_train_device[:num_training]
    y_inc = y_train_device[:num_training]
    X_inc_sample = torch.empty(0, X_train_device.size(1)).float().to(device)
    y_inc_sample = torch.empty(0).to(device)
    
    H_inv = lr_hessian_inv(w_approx_stream, X_inc, y_inc, lam)
    I = torch.eye(X_train_device.size(1)).float().to(device)
    Gram_sample_inverse = (lambda_reg * I).inverse()
    
    delta_newton_norms_stream = []
    update_count_stream = 0
    
    print(f"Performing {num_removes} incremental data arrivals (Streaming)...")
    for i in range(num_removes):
        new_point_idx = num_training + i
        
        X_inc = torch.cat([X_inc, X_train_device[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train_device[new_point_idx].unsqueeze(0)])
        
        grad_i = lr_grad(w_approx_stream, X_train_device[new_point_idx].unsqueeze(0), 
                        y_train_device[new_point_idx].unsqueeze(0), lam)
        Delta = grad_i
        
        leverage_score = compute_online_leverage_score_incremental(
            X_train_device[new_point_idx], Gram_sample_inverse, X_inc_sample, epsilon, lambda_reg)
        sample_prob = compute_sampling_probability(leverage_score, d, beta, epsilon)
        
        if torch.rand(1).item() < sample_prob:
            update_count_stream += 1
            
            weighted_x = X_train_device[new_point_idx] / math.sqrt(sample_prob)
            X_inc_sample = torch.cat([X_inc_sample, weighted_x.unsqueeze(0)])
            y_inc_sample = torch.cat([y_inc_sample, y_train_device[new_point_idx].unsqueeze(0)])
            
            X_for_hessian = torch.cat([X_train_device[:num_training], X_inc_sample])
            y_for_hessian = torch.cat([y_train_device[:num_training], y_inc_sample])
            H_inv = lr_hessian_inv_approximate(w_approx_stream, X_for_hessian, X_inc, y_for_hessian, lam)
            
            Gram_sample_inverse = update_XTX_inverse(weighted_x, Gram_sample_inverse, lambda_reg)
            
            delta_newton = H_inv.mv(Delta)
            delta_newton_norms_stream.append(delta_newton.norm().item())
            w_approx_stream -= delta_newton
            
            Delta = torch.zeros_like(Delta)
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count_stream} updates...")
    
    # Calculate Streaming test accuracy
    pred_stream = X_test_device.mv(w_approx_stream)
    test_acc_stream = pred_stream.gt(0).squeeze().eq(y_test_device.gt(0)).float().mean().item()
    
    results['streaming'] = {
        'delta_newton_norms': delta_newton_norms_stream,
        'mean_norm': np.mean(delta_newton_norms_stream) if delta_newton_norms_stream else 0,
        'std_norm': np.std(delta_newton_norms_stream) if delta_newton_norms_stream else 0,
        'max_norm': np.max(delta_newton_norms_stream) if delta_newton_norms_stream else 0,
        'min_norm': np.min(delta_newton_norms_stream) if delta_newton_norms_stream else 0,
        'sum_norm': np.sum(delta_newton_norms_stream) if delta_newton_norms_stream else 0,
        'num_updates': update_count_stream,
        'test_accuracy': test_acc_stream
    }
    
    print(f"\nStreaming Results:")
    print(f"  Number of updates: {update_count_stream}")
    print(f"  Test accuracy: {test_acc_stream:.4f}")
    print(f"  Delta Newton mean norm: {results['streaming']['mean_norm']:.6f}")
    print(f"  Delta Newton norm sum: {results['streaming']['sum_norm']:.6f}")
    
    # ===== Run Uniform Method =====
    print(f"\nRunning UNIFORM method (p={uniform_prob})...")
    torch.manual_seed(42)
    
    # Re-initialize
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx_uniform = w.clone()
    
    Delta = torch.zeros(d).to(device)
    X_inc = X_train_device[:num_training]
    y_inc = y_train_device[:num_training]
    X_inc_sample = torch.empty(0, X_train_device.size(1)).float().to(device)
    y_inc_sample = torch.empty(0).to(device)
    
    H_inv = lr_hessian_inv(w_approx_uniform, X_inc, y_inc, lam)
    
    delta_newton_norms_uniform = []
    update_count_uniform = 0
    
    print(f"Performing {num_removes} incremental data arrivals (Uniform)...")
    for i in range(num_removes):
        new_point_idx = num_training + i
        
        X_inc = torch.cat([X_inc, X_train_device[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train_device[new_point_idx].unsqueeze(0)])
        
        grad_i = lr_grad(w_approx_uniform, X_train_device[new_point_idx].unsqueeze(0),
                        y_train_device[new_point_idx].unsqueeze(0), lam)
        Delta = grad_i
        
        if torch.rand(1).item() < uniform_prob:
            update_count_uniform += 1
            
            X_inc_sample = torch.cat([X_inc_sample, X_train_device[new_point_idx].unsqueeze(0) / math.sqrt(uniform_prob)])
            y_inc_sample = torch.cat([y_inc_sample, y_train_device[new_point_idx].unsqueeze(0)])
            
            H_inv = lr_hessian_inv_approximate(w_approx_uniform, X_inc_sample, X_inc, y_inc_sample, lam)
            
            delta_newton = H_inv.mv(Delta)
            delta_newton_norms_uniform.append(delta_newton.norm().item())
            w_approx_uniform -= delta_newton
            
            Delta = torch.zeros_like(Delta)
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count_uniform} updates...")
    
    # Calculate Uniform test accuracy
    pred_uniform = X_test_device.mv(w_approx_uniform)
    test_acc_uniform = pred_uniform.gt(0).squeeze().eq(y_test_device.gt(0)).float().mean().item()
    
    results['uniform'] = {
        'delta_newton_norms': delta_newton_norms_uniform,
        'mean_norm': np.mean(delta_newton_norms_uniform) if delta_newton_norms_uniform else 0,
        'std_norm': np.std(delta_newton_norms_uniform) if delta_newton_norms_uniform else 0,
        'max_norm': np.max(delta_newton_norms_uniform) if delta_newton_norms_uniform else 0,
        'min_norm': np.min(delta_newton_norms_uniform) if delta_newton_norms_uniform else 0,
        'sum_norm': np.sum(delta_newton_norms_uniform) if delta_newton_norms_uniform else 0,
        'num_updates': update_count_uniform,
        'test_accuracy': test_acc_uniform
    }
    
    print(f"\nUniform Results:")
    print(f"  Number of updates: {update_count_uniform}")
    print(f"  Test accuracy: {test_acc_uniform:.4f}")
    print(f"  Delta Newton mean norm: {results['uniform']['mean_norm']:.6f}")
    print(f"  Delta Newton norm sum: {results['uniform']['sum_norm']:.6f}")
    
    # ===== Run Incremental Method (p=1, full sampling) =====
    print(f"\nRunning INCREMENTAL method (p=1.0, full sampling)...")
    torch.manual_seed(42)
    
    # Re-initialize
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx_incr = w.clone()
    
    X_inc = X_train_device[:num_training]
    y_inc = y_train_device[:num_training]
    
    H_inv = lr_hessian_inv(w_approx_incr, X_inc, y_inc, lam)
    
    delta_newton_norms_incr = []
    update_count_incr = 0
    
    print(f"Performing {num_removes} incremental data arrivals (Incremental)...")
    for i in range(num_removes):
        new_point_idx = num_training + i
        
        X_inc = torch.cat([X_inc, X_train_device[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train_device[new_point_idx].unsqueeze(0)])
        
        grad_i = lr_grad(w_approx_incr, X_train_device[new_point_idx].unsqueeze(0),
                        y_train_device[new_point_idx].unsqueeze(0), lam)
        
        # Incremental method: update for every point
        update_count_incr += 1
        
        H_inv = lr_hessian_inv(w_approx_incr, X_inc, y_inc, lam)
        
        delta_newton = H_inv.mv(grad_i)
        delta_newton_norms_incr.append(delta_newton.norm().item())
        w_approx_incr -= delta_newton
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count_incr} updates...")
    
    # Calculate Incremental test accuracy
    pred_incr = X_test_device.mv(w_approx_incr)
    test_acc_incr = pred_incr.gt(0).squeeze().eq(y_test_device.gt(0)).float().mean().item()
    
    results['incremental'] = {
        'delta_newton_norms': delta_newton_norms_incr,
        'mean_norm': np.mean(delta_newton_norms_incr) if delta_newton_norms_incr else 0,
        'std_norm': np.std(delta_newton_norms_incr) if delta_newton_norms_incr else 0,
        'max_norm': np.max(delta_newton_norms_incr) if delta_newton_norms_incr else 0,
        'min_norm': np.min(delta_newton_norms_incr) if delta_newton_norms_incr else 0,
        'sum_norm': np.sum(delta_newton_norms_incr) if delta_newton_norms_incr else 0,
        'num_updates': update_count_incr,
        'test_accuracy': test_acc_incr
    }
    
    print(f"\nIncremental Results:")
    print(f"  Number of updates: {update_count_incr}")
    print(f"  Test accuracy: {test_acc_incr:.4f}")
    print(f"  Delta Newton mean norm: {results['incremental']['mean_norm']:.6f}")
    print(f"  Delta Newton norm sum: {results['incremental']['sum_norm']:.6f}")
    
    return results


def plot_delta_newton_comparison(results, dataset_name, output_dir):
    """
    Plot box plots of delta_newton norms for Leverage, Uniform, and Incremental methods.
    
    Args:
        results: Return value from run_streaming_uniform_delta_newton_comparison
        dataset_name: Name of the dataset
        output_dir: Output directory
    """
    # Prepare data - delta_newton norms for all three methods
    data_to_plot = [
        results['streaming']['delta_newton_norms'],
        results['uniform']['delta_newton_norms'],
        results['incremental']['delta_newton_norms']
    ]
    
    methods = ['Leverage', 'Uniform', 'Incremental']
    
    # Create figure - box plots
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors: streaming (green), uniform (blue), incremental (red)
    colors = ['#388E3C', '#1976D2', '#E63946']
    
    # Draw box plots (without outliers)
    bp = ax.boxplot(data_to_plot, 
                    labels=methods,
                    patch_artist=True,  # Fill colors
                    widths=0.6,
                    showmeans=True,  # Show means
                    showfliers=False,  # Don't show outliers
                    meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8))
    
    # Set colors for each box
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    # Make median lines dark blue and thick
    plt.setp(bp['medians'], color='darkblue', linewidth=2)
    
    plt.ylabel('Norm of the Incremental Update', 
               fontsize=16, fontweight='bold')
    ax.set_title(f'Delta Newton Norm Distribution Comparison\n{dataset_name} Dataset',
                 fontsize=18, fontweight='bold', pad=20)
    
    # Optimize axis tick display
    ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
    ax.tick_params(axis='both', which='minor', labelsize=12, width=1.5, length=4)
    ax.tick_params(axis='x', rotation=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add legend explaining box plot elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.7, edgecolor='black', label='Leverage'),
        Patch(facecolor=colors[1], alpha=0.7, edgecolor='black', label='Uniform'),
        Patch(facecolor=colors[2], alpha=0.7, edgecolor='black', label='Incremental'),
        plt.Line2D([0], [0], color='darkblue', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
                   markeredgecolor='red', markersize=8, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
              framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f'delta_newton_comparison_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Delta Newton comparison plot saved to: {png_path}")
    
    # Save PDF
    pdf_path = png_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"Delta Newton comparison plot (PDF) saved to: {pdf_path}")
    
    plt.close()


def save_delta_newton_comparison_to_csv(results, dataset_name, output_dir):
    """
    Save delta_newton comparison results to CSV.
    
    Args:
        results: Return value from run_streaming_uniform_delta_newton_comparison
        dataset_name: Name of the dataset
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save statistical summary
    summary_path = os.path.join(output_dir, f'delta_newton_comparison_summary_{dataset_name.lower().replace(" ", "_")}.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'num_updates', 'test_accuracy', 'mean_norm', 'std_norm', 
                     'max_norm', 'min_norm', 'sum_norm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for method_name in ['streaming', 'uniform', 'incremental']:
            if method_name == 'streaming':
                method_label = 'Streaming (Leverage)'
            elif method_name == 'uniform':
                method_label = 'Uniform Sampling'
            else:
                method_label = 'Incremental (Full)'
            
            writer.writerow({
                'method': method_label,
                'num_updates': results[method_name]['num_updates'],
                'test_accuracy': results[method_name]['test_accuracy'],
                'mean_norm': results[method_name]['mean_norm'],
                'std_norm': results[method_name]['std_norm'],
                'max_norm': results[method_name]['max_norm'],
                'min_norm': results[method_name]['min_norm'],
                'sum_norm': results[method_name]['sum_norm']
            })
    
    print(f"Delta Newton comparison summary saved to: {summary_path}")
    
    # Save detailed delta_newton norm sequences
    for method_name in ['streaming', 'uniform', 'incremental']:
        method_label = method_name
        detail_path = os.path.join(output_dir, 
                                   f'delta_newton_{method_label}_{dataset_name.lower().replace(" ", "_")}.csv')
        with open(detail_path, 'w', newline='') as csvfile:
            fieldnames = ['update_index', 'delta_newton_norm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            norms = results[method_name]['delta_newton_norms']
            for i, norm in enumerate(norms):
                writer.writerow({
                    'update_index': i + 1,
                    'delta_newton_norm': norm
                })
        
        print(f"  {method_label.capitalize()} detailed data saved to: {detail_path}")


def run_delta_newton_experiment(beta=1.02, lambda_reg=0.014, uniform_prob=0.2, data_dir='./data'):
    """
    Run Delta Newton norm comparison experiment.
    
    Args:
        beta: Beta parameter for Streaming method
        lambda_reg: Lambda regularization parameter for Streaming method
        uniform_prob: Sampling probability for Uniform method
        data_dir: Data directory
    """
    print("="*80)
    print("DELTA NEWTON Norm Comparison Experiment - MNIST")
    print("="*80)
    print(f"Streaming parameters: β={beta}, λ={lambda_reg}")
    print(f"Uniform parameters: p={uniform_prob}")
    
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
    
    # Prepare arguments
    args = {**dataset_config, **GRADIENT_NORM_ARGS}
    
    # Set specific std value for Delta Newton experiment
    args['std'] = 10
    args['lam'] = 1e-3
    
    print(f"\nExperiment configuration:")
    print(f"  Base training samples: {args['num_training']}")
    print(f"  Incremental samples: {args['num_removes']}")
    print(f"  Lambda (regularization): {args['lam']}")
    print(f"  Std (perturbation): {args['std']}")
    
    # Run comparison experiment
    print("\n" + "="*80)
    print("Running STREAMING vs UNIFORM Delta Newton Comparison")
    print("="*80)
    results = run_streaming_uniform_delta_newton_comparison(
        X_train, y_train, X_test, y_test, args, beta, lambda_reg, uniform_prob
    )
    
    # Create output directories
    csv_output_dir = os.path.join('results', 'csv', 'delta_newton_comparison')
    plot_output_dir = os.path.join('results', 'plots', 'delta_newton_comparison')
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    save_delta_newton_comparison_to_csv(results, dataset_name, csv_output_dir)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations")
    print("="*80)
    plot_delta_newton_comparison(results, dataset_name, plot_output_dir)
    
    print("\n" + "="*80)
    print("Experiment Completed")
    print("="*80)
    print(f"\nResults summary:")
    print(f"\nStreaming (Leverage):")
    print(f"  Number of updates: {results['streaming']['num_updates']}")
    print(f"  Test accuracy: {results['streaming']['test_accuracy']:.4f}")
    print(f"  Delta Newton mean norm: {results['streaming']['mean_norm']:.6f}")
    print(f"  Delta Newton norm range: [{results['streaming']['min_norm']:.6f}, {results['streaming']['max_norm']:.6f}]")
    print(f"  Delta Newton norm sum: {results['streaming']['sum_norm']:.4f}")
    print(f"\nUniform Sampling:")
    print(f"  Number of updates: {results['uniform']['num_updates']}")
    print(f"  Test accuracy: {results['uniform']['test_accuracy']:.4f}")
    print(f"  Delta Newton mean norm: {results['uniform']['mean_norm']:.6f}")
    print(f"  Delta Newton norm range: [{results['uniform']['min_norm']:.6f}, {results['uniform']['max_norm']:.6f}]")
    print(f"  Delta Newton norm sum: {results['uniform']['sum_norm']:.4f}")
    print(f"\nIncremental (Full):")
    print(f"  Number of updates: {results['incremental']['num_updates']}")
    print(f"  Test accuracy: {results['incremental']['test_accuracy']:.4f}")
    print(f"  Delta Newton mean norm: {results['incremental']['mean_norm']:.6f}")
    print(f"  Delta Newton norm range: [{results['incremental']['min_norm']:.6f}, {results['incremental']['max_norm']:.6f}]")
    print(f"  Delta Newton norm sum: {results['incremental']['sum_norm']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run sampling property analysis experiment')
    parser.add_argument('--mode', type=str, default='leverage', 
                       choices=['leverage', 'delta_newton'],
                       help='Experiment mode: leverage (default) or delta_newton comparison')
    parser.add_argument('--beta', type=float, default=1.02,
                       help='Beta parameter for streaming (default: 1.02)')
    parser.add_argument('--lambda-reg', type=float, default=0.014,
                       help='Lambda regularization for leverage score (default: 0.014)')
    parser.add_argument('--uniform-prob', type=float, default=0.2,
                       help='Uniform sampling probability for delta_newton mode (default: 0.2)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top leverage score samples to track (default: 5)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Data directory (default: ./data)')
    
    args = parser.parse_args()
    
    if args.mode == 'leverage':
        run_experiment(
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            top_k=args.top_k,
            data_dir=args.data_dir
        )
    elif args.mode == 'delta_newton':
        run_delta_newton_experiment(
            beta=args.beta,
            lambda_reg=args.lambda_reg,
            uniform_prob=args.uniform_prob,
            data_dir=args.data_dir
        )


if __name__ == '__main__':
    main()

