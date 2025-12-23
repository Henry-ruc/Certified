# Core functions shared across all methods
# Extracted from Streaming_Learn.py, Uni_Sampling_Learn.py, and Incremental_Learn.py

import math
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_loss(w, X, y, lam):
    """Logistic regression loss with L2 regularization"""
    return -F.logsigmoid(y * X.mv(w)).mean() + lam * w.pow(2).sum() / 2


def lr_grad(w, X, y, lam):
    """Gradient of logistic regression loss"""
    z = torch.sigmoid(y * X.mv(w))
    return X.t().mv((z-1) * y) + lam * X.size(0) * w


def lr_hessian_inv(w, X, y, lam, batch_size=50000):
    """Compute inverse Hessian matrix for logistic regression"""
    z = torch.sigmoid(X.mv(w).mul_(y))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X.size(0) / batch_size))
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X.size(0))
        X_i = X[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    return (H + lam * X.size(0) * torch.eye(X.size(1)).float().to(device)).inverse()


def lr_hessian_inv_approximate(w, X_sample, X_original, y_sample, lam, batch_size=50000):
    """
    Compute approximate Hessian inverse using sampled matrix but with original matrix size for regularization.
    
    Args:
        w (torch.Tensor): Model parameters
        X_sample (torch.Tensor): Sampled feature matrix
        X_original (torch.Tensor): Original feature matrix (for size reference)
        y_sample (torch.Tensor): Sampled labels
        lam (float): L2 regularization coefficient
        batch_size (int): Batch size for computation
    
    Returns:
        torch.Tensor: Approximate inverse Hessian matrix
    """
    # Compute sigmoid values using sampled data
    z = torch.sigmoid(X_sample.mv(w).mul_(y_sample))
    D = z * (1 - z)
    H = None
    num_batch = int(math.ceil(X_sample.size(0) / batch_size))
    # Compute Hessian using sampled data
    for i in range(num_batch):
        lower = i * batch_size
        upper = min((i+1) * batch_size, X_sample.size(0))
        X_i = X_sample[lower:upper]
        if H is None:
            H = X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
        else:
            H += X_i.t().mm(D[lower:upper].unsqueeze(1) * X_i)
    # Use original matrix size for regularization
    n_original = X_original.size(0)
    d = X_original.size(1)
    # Add regularization term using original matrix size
    return (H + lam * n_original * torch.eye(d).float().to(device)).inverse()


def lr_optimize(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    """Optimize logistic regression using L-BFGS"""
    w = torch.autograd.Variable(torch.zeros(X.size(1)).float().to(device), requires_grad=True)
    def closure():
        if b is None:
            return lr_loss(w, X, y, lam)
        else:
            return lr_loss(w, X, y, lam) + b.dot(w) / X.size(0)
    optimizer = optim.LBFGS([w], tolerance_grad=tol, tolerance_change=1e-20)
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        loss.backward()
        if verbose and i == num_steps - 1:  # Only print last iteration
            print('Iteration %d: loss = %.6f, grad_norm = %.6f' % (i+1, loss.cpu(), w.grad.norm()))
        optimizer.step(closure)
    return w.data


def lr_optimize_lbfgs_torchmin(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    """Optimize logistic regression using L-BFGS (torchmin implementation)"""
    try:
        from torchmin import minimize
    except ImportError:
        raise ImportError("pytorch-minimize (torchmin) is required for L-BFGS optimization. Please install it first.")
    
    # Initial parameters
    w0 = torch.zeros(X.size(1)).float().to(device)
    
    # Define objective function
    def objective(w):
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        return loss
    
    # Run L-BFGS optimization
    result = minimize(
        objective, 
        w0, 
        method='l-bfgs',
        max_iter=num_steps,
        tol=tol,
        disp=2 if verbose else 0
    )
    
    return result.x


def lr_optimize_bfgs(X, y, lam, b=None, num_steps=100, tol=1e-10, verbose=False):
    """Optimize logistic regression using BFGS (torchmin implementation)"""
    try:
        from torchmin import minimize
    except ImportError:
        raise ImportError("pytorch-minimize (torchmin) is required for BFGS optimization. Please install it first.")
    
    # Initial parameters
    w0 = torch.zeros(X.size(1)).float().to(device)
    
    # Define objective function
    def objective(w):
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        return loss
    
    # Run BFGS optimization
    result = minimize(
        objective, 
        w0, 
        method='bfgs',
        max_iter=num_steps,
        tol=tol,
        disp=2 if verbose else 0
    )
    
    return result.x


def lr_optimize_newton(X, y, lam, b=None, num_steps=100, lr=1.0, verbose=False):
    """Optimize logistic regression using Newton's method (torchmin newton-exact)"""
    try:
        from torchmin import minimize
    except ImportError:
        raise ImportError("pytorch-minimize (torchmin) is required for Newton optimization. Please install it first.")
    
    # Initial parameters
    w0 = torch.zeros(X.size(1)).float().to(device)
    
    # Define objective function
    def objective(w):
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        return loss
    
    # Run Newton-exact optimization
    result = minimize(
        objective, 
        w0, 
        method='newton-exact',
        max_iter=num_steps,
        disp=2 if verbose else 0
    )
    
    return result.x


def lr_optimize_trust_exact(X, y, lam, b=None, num_steps=100, verbose=False):
    """Optimize logistic regression using Trust-Region Exact method (torchmin implementation)"""
    try:
        from torchmin import minimize
    except ImportError:
        raise ImportError("pytorch-minimize (torchmin) is required for Trust-Region Exact optimization. Please install it first.")
    
    # Initial parameters
    w0 = torch.zeros(X.size(1)).float().to(device)
    
    # Define objective function
    def objective(w):
        loss = lr_loss(w, X, y, lam)
        if b is not None:
            loss += b.dot(w) / X.size(0)
        return loss
    
    # Run Trust-Region Exact optimization
    result = minimize(
        objective, 
        w0, 
        method='trust-exact',
        max_iter=num_steps,
        disp=2 if verbose else 0
    )
    
    return result.x


def batch_multiply(A, B, batch_size=500000):
    """Batch matrix multiplication for large matrices"""
    if A.is_cuda:
        if len(B.size()) == 1:
            return A.mv(B)
        else:
            return A.mm(B)
    else:
        out = []
        num_batch = int(math.ceil(A.size(0) / float(batch_size)))
        with torch.no_grad():
            for i in range(num_batch):
                lower = i * batch_size
                upper = min((i+1) * batch_size, A.size(0))
                A_sub = A[lower:upper]
                A_sub = A_sub.to(device)
                if len(B.size()) == 1:
                    out.append(A_sub.mv(B).cpu())
                else:
                    out.append(A_sub.mm(B).cpu())
        return torch.cat(out, dim=0).to(device)


def spectral_norm(A, num_iters=20):
    """Compute spectral norm using power iteration"""
    x = torch.randn(A.size(0)).float().to(device)
    norm = 1
    for i in range(num_iters):
        x = A.mv(x)
        norm = x.norm()
        x /= norm
    return math.sqrt(norm)


def compute_online_leverage_score_incremental(X_new, XTX_inv_prev, X_curr, epsilon=0.5, lambda_reg=0.1):
    """
    Incremental version using Sherman-Morrison formula for rank-1 updates
    
    Args:
        X_new: New data point
        XTX_inv_prev: Previous (X^T X)^{-1} matrix
        X_curr: Current data matrix (unused, for API compatibility)
        epsilon: Error parameter
        lambda_reg: Regularization parameter
    """
    x = X_new.view(-1)
    
    # Compute x^T (X^T X)^{-1} x using previous inverse
    raw_value = torch.matmul(torch.matmul(x, XTX_inv_prev), x)
    # With float64, raw_value should always be non-negative (positive definite)
    # Still add safety check for robustness
    leverage_score = min(1.0, (1 + epsilon) * max(0.0, raw_value))
    if isinstance(leverage_score, torch.Tensor):
        return leverage_score.item()
    else:
        return float(leverage_score)


def update_XTX_inverse(X_new, XTX_inv_prev, lambda_reg=0.1):
    """
    Update (X^T X + lambda_reg * I)^{-1} when adding a new data point using Sherman-Morrison
    
    Args:
        X_new: New data point to add
        XTX_inv_prev: Previous (X^T X)^{-1} matrix
        lambda_reg: Regularization parameter (unused, for API compatibility)
    """
    x = X_new.view(-1, 1)  # Column vector
    
    # Sherman-Morrison formula: (A + xx^T)^{-1} = A^{-1} - (A^{-1}xx^T A^{-1})/(1 + x^T A^{-1} x)
    numerator = torch.matmul(torch.matmul(XTX_inv_prev, x), torch.matmul(x.t(), XTX_inv_prev))
    denominator = 1 + torch.matmul(torch.matmul(x.t(), XTX_inv_prev), x)
    
    XTX_inv_new = XTX_inv_prev - numerator / denominator
    
    return XTX_inv_new


def compute_sampling_probability(leverage_score, d, beta=1.0, epsilon=0.5):
    """
    Compute sampling probability based on leverage score.
    
    Args:
        leverage_score (float): Leverage score of the data point
        d (int): Feature dimension (unused, for API compatibility)
        beta (float): Sampling rate parameter
        epsilon (float): Error parameter (unused, for API compatibility)
    
    Returns:
        float: Sampling probability between 0 and 1
    """
    return min(1.0, beta * leverage_score)


def compute_approximate_ridge_leverage_scores(X, w, y, lam, r=None):
    """
    Compute approximate ridge leverage scores for subsampled Newton method.
    Uses exact diagonal leverage scores based on current Hessian.
    
    Args:
        X (torch.Tensor): Feature matrix (n x d)
        w (torch.Tensor): Current model parameters
        y (torch.Tensor): Labels
        lam (float): L2 regularization coefficient
        r (int): Unused, kept for API compatibility
    
    Returns:
        torch.Tensor: Approximate leverage scores for each data point
    """
    n, d = X.shape
    
    # Compute Hessian diagonal D^2 = sigmoid(y*X*w) * (1 - sigmoid(y*X*w))
    z = torch.sigmoid(y * X.mv(w))
    D2 = z * (1 - z)
    D = torch.sqrt(D2 + 1e-10)
    
    # Compute diagonal leverage scores efficiently
    # lev[i] = x_i^T (X^T D^2 X + lambda*n*I)^{-1} x_i * D_i^2
    # Approximate using: lev ~ D^2 * diag(X * (X^T D^2 X + lambda*n*I)^{-1} * X^T)
    
    # For efficiency, use row norm approximation weighted by D^2
    # This is similar to the RnormSS method in SSN
    row_norms_sq = (X ** 2).sum(dim=1)
    lev = D2 * row_norms_sq
    
    # Ensure non-negative
    lev = torch.clamp(lev, min=1e-10)
    
    return lev


def lr_optimize_subsampled_newton_lev(X, y, lam, b=None, num_steps=100, 
                                       hessian_size=None, mh=10, verbose=False):
    """
    Optimize logistic regression using subsampled Newton method with leverage score sampling.
    
    This method uses approximate leverage scores to intelligently sample data points for 
    computing an approximate Hessian, while using the full gradient. This typically 
    requires fewer iterations than first-order methods while being more efficient than 
    exact Newton's method.
    
    Args:
        X (torch.Tensor): Feature matrix (n x d)
        y (torch.Tensor): Labels
        lam (float): L2 regularization coefficient
        b (torch.Tensor, optional): Additional linear term in objective
        num_steps (int): Number of Newton iterations
        hessian_size (int, optional): Expected number of samples for Hessian approximation 
                                     (default: min(5*d, n))
        mh (int): Hessian approximation frequency - recompute leverage scores every mh iterations
        verbose (bool): Whether to print iteration information
    
    Returns:
        torch.Tensor: Optimized model parameters
    """
    n, d = X.shape
    if hessian_size is None:
        hessian_size = min(5 * d, n)  # Default: 5*d samples
    
    w = torch.zeros(d, device=device)
    eta = 1.0  # Step size
    
    # Pre-compute row norms for faster leverage score computation
    row_norms_sq = (X ** 2).sum(dim=1)
    
    # Cache for regularization term
    lam_n_I = lam * n * torch.eye(d, device=device)
    
    if verbose:
        print("Subsampled Newton (Leverage Score) solver starts...")
    
    for i in range(num_steps):
        # Recompute leverage scores every mh iterations
        if i % mh == 0:
            # Compute Hessian diagonal for weighting
            z = torch.sigmoid(y * X.mv(w))
            D2 = z * (1 - z)
            lev = D2 * row_norms_sq
            lev = torch.clamp(lev, min=1e-10)
            
            # Compute sampling probabilities
            p0 = lev / lev.sum()
            q = p0 * hessian_size
            q = torch.clamp(q, max=1.0)
        
        # Sample data points based on probabilities  
        rand_vals = torch.rand(n, device=device)
        idx = rand_vals < q
        
        if idx.sum() == 0:
            idx[torch.randint(0, n, (1,), device=device)] = True
        
        p_sub = q[idx]
        X_sub = X[idx]
        y_sub = y[idx]
        
        # Compute subsampled Hessian - always recalculate for stability
        z_sub = torch.sigmoid(y_sub * X_sub.mv(w))
        D2_sub = z_sub * (1 - z_sub)
        D2_weighted = D2_sub / p_sub
        
        H = X_sub.t().mm(D2_weighted.unsqueeze(1) * X_sub) + lam_n_I
        
        # Compute full gradient - always recalculate for accuracy
        z_full = torch.sigmoid(y * X.mv(w))
        grad_full = X.t().mv((z_full - 1) * y) + lam * n * w
        
        if b is not None:
            grad_full += b
        
        # Compute search direction with robust error handling
        try:
            search_dir = torch.linalg.solve(H, -grad_full)
        except (RuntimeError, torch._C._LinAlgError):
            # Add stronger regularization if matrix is singular
            H_reg = H + max(1e-3, 100 * lam * n) * torch.eye(d, device=device)
            try:
                search_dir = torch.linalg.solve(H_reg, -grad_full)
            except (RuntimeError, torch._C._LinAlgError):
                # If still fails, use gradient descent step
                search_dir = -grad_full / (lam * n + 1e-6)
        
        # Update parameters
        w = w + eta * search_dir
        
        if verbose and (i % 10 == 0 or i == num_steps - 1):
            loss = lr_loss(w, X, y, lam)
            if b is not None:
                loss += b.dot(w) / n
            print(f'Iteration {i+1}: loss = {loss.item():.6f}, grad_norm = {grad_full.norm().item():.6f}')
    
    if verbose:
        print("Subsampled Newton (Leverage Score) solver ends")
    
    return w


def lr_optimize_subsampled_newton_uniform(X, y, lam, b=None, num_steps=100, 
                                          hessian_size=None, verbose=False):
    """
    Optimize logistic regression using subsampled Newton method with uniform sampling.
    
    This method uniformly samples data points for computing an approximate Hessian,
    while using the full gradient. This is simpler than leverage score sampling but
    may require more samples for the same accuracy.
    
    Args:
        X (torch.Tensor): Feature matrix (n x d)
        y (torch.Tensor): Labels
        lam (float): L2 regularization coefficient
        b (torch.Tensor, optional): Additional linear term in objective
        num_steps (int): Number of Newton iterations
        hessian_size (int, optional): Number of samples for Hessian approximation 
                                     (default: min(3*d, n))
        verbose (bool): Whether to print iteration information
    
    Returns:
        torch.Tensor: Optimized model parameters
    """
    n, d = X.shape
    if hessian_size is None:
        hessian_size = min(3 * d, n)  # Default: 3*d samples
    
    w = torch.zeros(d, device=device)
    eta = 1.0  # Step size
    
    # Cache for regularization term
    lam_n_I = lam * n * torch.eye(d, device=device)
    
    if verbose:
        print("Subsampled Newton (Uniform) solver starts...")
    
    for i in range(num_steps):
        # Uniformly sample data points (with replacement for consistency with SSN)
        idx = torch.randint(0, n, (hessian_size,), device=device)
        X_sub = X[idx]
        y_sub = y[idx]
        
        # Compute subsampled Hessian
        z_sub = torch.sigmoid(y_sub * X_sub.mv(w))
        D2_sub = z_sub * (1 - z_sub)
        # With replacement sampling: each point has probability 1/n, importance weight is n
        D2_weighted = D2_sub * n
        
        H = X_sub.t().mm(D2_weighted.unsqueeze(1) * X_sub) + lam_n_I
        
        # Compute full gradient
        z_full = torch.sigmoid(y * X.mv(w))
        grad_full = X.t().mv((z_full - 1) * y) + lam * n * w
        
        if b is not None:
            grad_full += b
        
        # Compute search direction with robust error handling
        try:
            search_dir = torch.linalg.solve(H, -grad_full)
        except (RuntimeError, torch._C._LinAlgError):
            # Add stronger regularization if matrix is singular
            H_reg = H + max(1e-3, 100 * lam * n) * torch.eye(d, device=device)
            try:
                search_dir = torch.linalg.solve(H_reg, -grad_full)
            except (RuntimeError, torch._C._LinAlgError):
                # If still fails, use gradient descent step
                search_dir = -grad_full / (lam * n + 1e-6)
        
        # Update parameters
        w = w + eta * search_dir
        
        if verbose and (i % 10 == 0 or i == num_steps - 1):
            loss = lr_loss(w, X, y, lam)
            if b is not None:
                loss += b.dot(w) / n
            print(f'Iteration {i+1}: loss = {loss.item():.6f}, grad_norm = {grad_full.norm().item():.6f}')
    
    if verbose:
        print("Subsampled Newton (Uniform) solver ends")
    
    return w

