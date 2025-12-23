# Streaming method runner (leverage score sampling)
# Based on Streaming_Learn.py

import time
import torch
import math
from core_functions import (
    lr_loss, lr_grad, lr_hessian_inv, lr_hessian_inv_approximate, lr_optimize,
    spectral_norm, compute_online_leverage_score_incremental, update_XTX_inverse,
    compute_sampling_probability, device
)


def run_streaming_experiment(X_train, y_train, X_test, y_test, args, beta, lambda_reg):
    """
    Run streaming learning with leverage score sampling.
    
    Args:
        X_train: Training data (base + update data)
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
        args: Dictionary with arguments (lam, std, num_training, num_removes, etc.)
        beta: Beta parameter for sampling probability
        lambda_reg: Lambda regularization parameter for leverage score
    
    Returns:
        tuple: (total_time, test_accuracy, num_samples)
    """
    print(f"\nRunning STREAMING method (β={beta}, λ={lambda_reg})...")
    
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
    b = std * torch.randn(X_train_base.size(1)).float().to(device)
    w = lr_optimize(X_train_base, y_train_base, lam, b=b, num_steps=num_steps, verbose=False)
    w_approx = w.clone()
    
    # Initialize for incremental updates
    # K = X_train_base.t().mm(X_train_base)  
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    
    # Track metrics
    # times = torch.zeros(num_removes).float()  
    update_count = 0  # Required: Used to track actual update count
    # min_sample_prob = float('inf')  
    
    start_all = time.time()
    
    for i in range(num_removes):
        # start = time.time()  
        
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
            # Initialize streaming covariance matrix K_streaming
            # K_streaming = torch.zeros_like(K)  
        
        # Original incremental data
        X_inc = torch.cat([X_inc, X_train[new_point_idx].unsqueeze(0)])
        y_inc = torch.cat([y_inc, y_train[new_point_idx].unsqueeze(0)])
        
        # Calculate gradient on the newly added data point
        grad_i = lr_grad(w_approx, X_train[new_point_idx].unsqueeze(0), y_train[new_point_idx].unsqueeze(0), lam)
        # Calculate Delta
        # Delta = Delta + grad_i
        Delta = grad_i
        
        # Calculate the sampling probability
        leverage_score = compute_online_leverage_score_incremental(
            X_train[new_point_idx], Gram_sample_inverse, X_inc_sample, epsilon, lambda_reg)
        sample_prob = compute_sampling_probability(leverage_score, d, beta, epsilon)
        
        # Sample point with probability sample_prob
        if torch.rand(1).item() < sample_prob:
            # Increment update counter (only when we actually update)
            update_count += 1
            
            # Update minimum sampling probability
            # if sample_prob < min_sample_prob:  
            #     min_sample_prob = sample_prob
            
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
            
            # Update covariance matrix K with new data point
            # K += torch.ger(X_train[new_point_idx], X_train[new_point_idx])  
            # K_streaming += torch.ger(X_train[new_point_idx], X_train[new_point_idx])  
            
            # Newton update
            delta_newton = H_inv.mv(Delta)
            w_approx -= delta_newton
            
            # Reset Delta to 0 vector
            Delta = torch.zeros_like(Delta)
        
        # times[i] = time.time() - start  
        
        if (i + 1) % 2000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count} updates...")
    
    total_time = time.time() - start_all
    
    # Evaluate on test set
    pred = X_test.mv(w_approx)
    test_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    # Calculate number of samples (base + sampled streaming)
    X_full_hessian = torch.cat([X_train[:num_training], X_inc_sample])
    num_samples = len(X_full_hessian)
    
    print(f"  Time: {total_time:.2f}s | Accuracy: {test_accuracy:.4f} | Samples: {num_samples} | Updates: {update_count}")
    
    return total_time, test_accuracy, num_samples


def run_streaming_with_grad_norm(X_train, y_train, X_test, y_test, args, beta, lambda_reg):
    """
    Run streaming learning with leverage score sampling and gradient norm tracking.
    
    Similar to run_streaming_experiment but includes:
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
        beta: Beta parameter for sampling probability
        lambda_reg: Lambda regularization parameter for leverage score
    
    Returns:
        dict: {
            'total_time': Total computation time,
            'test_accuracy': Test set accuracy,
            'num_samples': Total samples used,
            'num_updates': Number of actual model updates,
            'beta': Beta parameter,
            'lambda_reg': Lambda regularization,
            'min_sample_prob': Minimum sampling probability among sampled points (None if no updates),
            'grad_norm_approx': Array of data-dependent gradient norm approximations per iteration,
            'worst_case_bound_approx': Array of worst-case bound per iteration,
            'true_grad_norm_per_iter': Array of true gradient norm per iteration,
            'sum_grad_norm': Cumulative data-dependent gradient norm bound,
            'sum_worst_case_bound': Cumulative worst-case bound,
            'sum_true_grad_norm': Cumulative true gradient norm,
            'times': Array of per-iteration times
        }
    """
    print(f"\n{'='*60}")
    print(f"Running STREAMING method with gradient norm tracking (β={beta}, λ={lambda_reg})")
    print(f"{'='*60}")
    
    # Extract parameters
    lam = args['lam']
    std = args['std']
    num_training = args['num_training']
    num_removes = args['num_removes']
    num_steps = args['num_steps']
    Lip_constant = args.get('lip_constant', 1/4)
    Gradnorm_constant = args.get('gradnorm_constant', 1)
    Weight_constant = args.get('weight_constant', 1)
    Hessian_constant = args.get('hessian_constant', 1/4)
    epsilon_hessian = args.get('epsilon_leverage', 0.1)
    
    # Calculate epsilon based on lambda_reg (delta / lambda_reg)
    delta = 0.01
    epsilon = delta / lambda_reg
    
    print(f"Using Lipschitz constant: {Lip_constant}")
    print(f"Using Gradient norm constant: {Gradnorm_constant}")
    print(f"Using Weight constant: {Weight_constant}")
    print(f"Using Hessian constant: {Hessian_constant}")
    print(f"Using epsilon_hessian (for Hessian approximation): {epsilon_hessian}")
    print(f"Using epsilon (for leverage score): {epsilon}")
    print(f"Beta: {beta}, Lambda_reg: {lambda_reg}")
    
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
    d = X_train.shape[1]
    Delta = torch.zeros(d).to(device)
    
    # Track time and gradient norms
    times = torch.zeros(num_removes).float()
    grad_norm_approx = torch.zeros(num_removes).float()
    worst_case_bound_approx = torch.zeros(num_removes).float()
    true_grad_norm_per_iter = torch.zeros(num_removes).float()
    sum_grad_norm = 0.0
    sum_worst_case_bound = 0.0
    sum_true_grad_norm = 0.0
    update_count = 0
    min_sample_prob = 100000 # Track minimum sampling probability among sampled points
    
    print(f"Performing {num_removes} data arrivals with leverage score sampling...")
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
        
        # Initialize term values for printing
        current_term1, current_term2, current_term3 = 0.0, 0.0, 0.0
        
        if sampled:
            # Increment update counter
            update_count += 1
            
            # Update minimum sampling probability
            if sample_prob < min_sample_prob:
                min_sample_prob = sample_prob
            
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
            
            # Update covariance matrix K with new data point
            K += torch.ger(X_train[new_point_idx], X_train[new_point_idx])
            spec_norm = spectral_norm(K)
            
            # Newton update
            delta_newton = H_inv.mv(Delta)
            Delta_p = X_for_hessian.mv(delta_newton)
            w_approx -= delta_newton
            
            
            # Compute spectral norm squared of incremental sampled data (excluding base)
            if len(X_inc_sample) > 0:
                K_inc_sample = X_inc_sample.t().mm(X_inc_sample)
                spec_norm_inc_sample = spectral_norm(K_inc_sample)
                spec_norm_inc_sample_sq = spec_norm_inc_sample 
            else:
                spec_norm_inc_sample_sq = 0.0
            
            # Compute data-dependent gradient norm approximation with three terms
            # Use min_sample_prob for tighter bound
            term1 = delta_newton.norm() * Delta_p.norm() * spec_norm * Lip_constant / math.sqrt(min_sample_prob)
            term2 = spec_norm * Delta_p.norm() * Weight_constant * Lip_constant * (1 / math.sqrt(min_sample_prob) - 1)
            term3 = epsilon_hessian * spec_norm_inc_sample_sq * Hessian_constant * delta_newton.norm() / (1-epsilon_hessian)
            grad_norm_approx[i] = (term1 + term2 + term3).cpu()
            sum_grad_norm += grad_norm_approx[i].item()
            
            # Store term values for printing
            current_term1 = term1.item() if torch.is_tensor(term1) else term1
            current_term2 = term2.item() if torch.is_tensor(term2) else term2
            current_term3 = term3.item() if torch.is_tensor(term3) else term3
            
            # Compute worst-case bound
            # Use min_sample_prob for tighter bound
            current_data_size = len(X_inc)  # All data seen (not just sampled)
            current_data_size_sampled = len(X_for_hessian)
            incremental_sample_size = current_data_size_sampled - num_training
            term1_wc = 4 * Lip_constant * (Gradnorm_constant ** 2) * current_data_size_sampled / ((lam ** 2) * (current_data_size ** 2) * (min_sample_prob ** 1.5))
            term2_wc = 2 * incremental_sample_size * Lip_constant * Weight_constant * Gradnorm_constant * (1 / math.sqrt(min_sample_prob) - 1) / (lam * current_data_size * min_sample_prob)
            term3_wc = 2 * epsilon_hessian * Gradnorm_constant * Hessian_constant / lam
            worst_case_bound_approx[i] = term1_wc + term2_wc + term3_wc
            sum_worst_case_bound += worst_case_bound_approx[i]
            
            # Reset Delta
            Delta = torch.zeros_like(Delta)
        else:
            # Not sampled: use Gradnorm_constant as approximation for both bounds
            # grad_norm_approx[i] = Gradnorm_constant
            # sum_grad_norm += grad_norm_approx[i]
            grad_norm_approx[i] = grad_i.norm()
            sum_grad_norm += grad_norm_approx[i]
            
            # worst_case_bound_approx[i] = Gradnorm_constant
            # sum_worst_case_bound += worst_case_bound_approx[i]
            worst_case_bound_approx[i] = grad_i.norm()
            sum_worst_case_bound += worst_case_bound_approx[i]
        
        # Use autograd to compute true gradient norm on all incremental data
        # This is computed regardless of whether the point was sampled or not
        w_tensor = torch.nn.Parameter(w_approx.clone())
        loss = lr_loss(w_tensor, X_inc, y_inc, lam)
        if b is not None:
            loss += b.dot(w_tensor) / X_inc.size(0)
        loss.backward()
        true_grad_norm_per_iter[i] = w_tensor.grad.norm().item()
        sum_true_grad_norm += true_grad_norm_per_iter[i]
        
        times[i] = time.time() - start
        
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i+1}/{num_removes} data arrivals, {update_count} updates...")
            if update_count > 0:
                print(f"    Min sampling probability: {min_sample_prob:.6f}")
            if sampled:
                print(f"    Current sampling probability: {sample_prob:.6f}")
                print(f"    Data-dependent grad norm bound: {grad_norm_approx[i]:.6f}")
                print(f"      - Term1 (Newton step): {current_term1:.6f}")
                print(f"      - Term2 (Weight correction): {current_term2:.6f}")
                print(f"      - Term3 (Hessian approximation): {current_term3:.6f}")
                print(f"    Worst-case bound: {worst_case_bound_approx[i]:.6f}")
            print(f"    Cumulative data-dependent bound: {sum_grad_norm:.6f}")
            print(f"    Cumulative worst-case bound: {sum_worst_case_bound:.6f}")
            print(f"    Cumulative true grad norm: {sum_true_grad_norm:.6f}")
    
    total_time = time.time() - start_all
    
    # Evaluate on test set
    pred = X_test.mv(w_approx)
    test_accuracy = pred.gt(0).squeeze().eq(y_test.gt(0)).float().mean().item()
    
    # Calculate number of samples (base + sampled streaming)
    X_full_hessian = torch.cat([X_train[:num_training], X_inc_sample])
    num_samples = len(X_full_hessian) - num_training
    
    print(f"\nStreaming Sampling Results with Gradient Norm Tracking:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Total samples used (weighted): {num_samples}")
    print(f"  Actual updates: {update_count}/{num_removes} ({100*update_count/num_removes:.1f}%)")
    if update_count > 0:
        print(f"  Minimum sampling probability: {min_sample_prob:.6f}")
    print(f"  Final cumulative data-dependent bound: {sum_grad_norm:.6f}")
    print(f"    Note: Data-dependent bound = Term1 + Term2 + Term3")
    print(f"      Term1: Newton step impact (||δ|| * ||Δ'|| * σ(K) * L / √p)")
    print(f"      Term2: Weight correction ((1/√p - 1) term)")
    print(f"      Term3: Hessian approximation error")
    print(f"  Final cumulative worst-case bound: {sum_worst_case_bound:.6f}")
    print(f"  Final cumulative true grad norm: {sum_true_grad_norm:.6f}")
    
    return {
        'total_time': total_time,
        'test_accuracy': test_accuracy,
        'num_samples': num_samples,
        'num_updates': update_count,
        'beta': beta,
        'lambda_reg': lambda_reg,
        'min_sample_prob': min_sample_prob if update_count > 0 else None,
        'grad_norm_approx': grad_norm_approx,
        'worst_case_bound_approx': worst_case_bound_approx,
        'true_grad_norm_per_iter': true_grad_norm_per_iter,
        'sum_grad_norm': sum_grad_norm,
        'sum_worst_case_bound': sum_worst_case_bound,
        'sum_true_grad_norm': sum_true_grad_norm,
        'times': times
    }
