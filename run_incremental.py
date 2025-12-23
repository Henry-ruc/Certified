# Incremental method runner (full updates, no sampling)
# Based on Incremental_Learn.py

import time
import torch
import math
from core_functions import (
    lr_loss, lr_grad, lr_hessian_inv, lr_optimize, 
    spectral_norm, device
)


def run_incremental_experiment(X_train, y_train, X_test, y_test, args):
    """
    Run incremental learning with full updates (no sampling).
    
    Args:
        X_train: Training data (base + update data)
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary with arguments (lam, std, num_training, num_removes, etc.)
    
    Returns:
        tuple: (total_time, test_accuracy, num_samples)
    """
    print("\n" + "="*60)
    print("Running INCREMENTAL method (full updates, baseline)")
    print("="*60)
    
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Split data
    n = X_train.size(0)
    num_inc_samples = n - num_training
    
    # Data already shuffled in data_loader, no need to shuffle again
    
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
    K = X_train_base.t().mm(X_train_base)
    
    # Track time
    times = torch.zeros(num_removes).float()
    
    print(f"Performing {num_removes} incremental updates...")
    start_all = time.time()
    
    for i in range(num_removes):
        start = time.time()
        
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
        
        times[i] = time.time() - start
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} updates...")
    
    total_time = time.time() - start_all
    
    # Evaluate on test set
    pred = X_test.mv(w_approx)
    test_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    num_samples = len(X_inc)  # Total samples used (base + all incremental)
    
    print(f"\nIncremental Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Total samples: {num_samples}")
    
    return total_time, test_accuracy, num_samples

def run_incremental_with_grad_norm(X_train, y_train, X_test, y_test, args):
    """
    Run incremental learning with full updates and gradient norm tracking.
    
    This function is similar to run_incremental_experiment but includes:
    - True gradient norm computation using autograd
    - Data-dependent gradient norm approximation tracking
    - Worst-case bound tracking
    - Detailed gradient norm statistics
    
    Args:
        X_train: Training data (base + update data)
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary with arguments (lam, std, num_training, num_removes, etc.)
    
    Returns:
        dict: {
            'total_time': Total computation time,
            'test_accuracy': Test set accuracy,
            'num_samples': Total samples used,
            'grad_norm_approx': Array of data-dependent gradient norm approximations per iteration,
            'worst_case_bound_approx': Array of worst-case bound per iteration,
            'true_grad_norm_per_iter': Array of true gradient norm per iteration,
            'sum_grad_norm': Cumulative data-dependent gradient norm bound,
            'sum_worst_case_bound': Cumulative worst-case bound,
            'sum_true_grad_norm': Cumulative true gradient norm,
            'times': Array of per-iteration times
        }
    """
    print("\n" + "="*60)
    print("Running INCREMENTAL method with gradient norm tracking")
    print("="*60)
    
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    Lip_constant = args.get('lip_constant', 1/4)  # Default to 1/4 if not provided
    Gradnorm_constant = args.get('gradnorm_constant', 1)  # Default to 1 if not provided
    
    print(f"Using Lipschitz constant: {Lip_constant}")
    print(f"Using Gradient norm constant: {Gradnorm_constant}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
    # Split data
    n = X_train.size(0)
    num_inc_samples = n - num_training
    
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
    K = X_train_base.t().mm(X_train_base)
    
    # Track time and gradient norms
    times = torch.zeros(num_removes).float()
    grad_norm_approx = torch.zeros(num_removes).float()
    worst_case_bound_approx = torch.zeros(num_removes).float()
    true_grad_norm_per_iter = torch.zeros(num_removes).float()  # Per-iteration true gradient norm
    sum_grad_norm = 0.0
    sum_worst_case_bound = 0.0
    sum_true_grad_norm = 0.0  # Cumulative true gradient norm
    
    print(f"Performing {num_removes} incremental updates with gradient norm tracking...")
    start_all = time.time()
    
    for i in range(num_removes):
        start = time.time()
        
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
        spec_norm = spectral_norm(K)
        
        # Newton update
        Delta = H_inv.mv(grad_i)
        Delta_p = X_inc.mv(Delta)
        w_approx -= Delta
        
        # Compute data-dependent gradient norm approximation
        grad_norm_approx[i] = (Delta.norm() * Delta_p.norm() * spec_norm * Lip_constant).cpu()
        sum_grad_norm += grad_norm_approx[i].item()

        # Compute worst-case bound
        current_data_size = len(X_inc)  # base + incremental samples so far
        worst_case_bound_approx[i] = 4 * Lip_constant * (Gradnorm_constant ** 2) / ((lam ** 2) * current_data_size)
        sum_worst_case_bound += worst_case_bound_approx[i]
        
        # Use autograd to compute true gradient norm on incremental data
        w_tensor = torch.nn.Parameter(w_approx.clone())
        loss = lr_loss(w_tensor, X_inc, y_inc, lam)
        if b is not None:
            loss += b.dot(w_tensor) / X_inc.size(0)
        loss.backward()
        true_grad_norm_per_iter[i] = w_tensor.grad.norm().item()
        sum_true_grad_norm += true_grad_norm_per_iter[i]
        
        times[i] = time.time() - start
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} updates...")
            print(f"    Data-dependent grad norm bound: {grad_norm_approx[i]:.6f}")
            print(f"    Worst-case bound: {worst_case_bound_approx[i]:.6f}")
            print(f"    Cumulative data-dependent bound: {sum_grad_norm:.6f}")
            print(f"    Cumulative worst-case bound: {sum_worst_case_bound:.6f}")
            print(f"    Cumulative true grad norm: {sum_true_grad_norm:.6f}")
    
    total_time = time.time() - start_all
    
    # Evaluate on test set
    pred = X_test.mv(w_approx)
    test_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    num_samples = len(X_inc)  # Total samples used (base + all incremental)
    
    print(f"\nIncremental Results with Gradient Norm Tracking:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Total samples: {num_samples}")
    print(f"  Final cumulative data-dependent bound: {sum_grad_norm:.6f}")
    print(f"  Final cumulative worst-case bound: {sum_worst_case_bound:.6f}")
    print(f"  Final cumulative true grad norm: {sum_true_grad_norm:.6f}")
    
    return {
        'total_time': total_time,
        'test_accuracy': test_accuracy,
        'num_samples': num_samples,
        'grad_norm_approx': grad_norm_approx,
        'worst_case_bound_approx': worst_case_bound_approx,
        'true_grad_norm_per_iter': true_grad_norm_per_iter,
        'sum_grad_norm': sum_grad_norm,
        'sum_worst_case_bound': sum_worst_case_bound,
        'sum_true_grad_norm': sum_true_grad_norm,
        'times': times
    }

