# Configuration parameters for pareto experiments
# Based on OPTIMAL_LEVERAGE_CONFIGURATIONS.md for CT Slice dataset

# Dataset configuration - CT Slice (Default)
DATASET_CONFIG = {
    'dataset': 'ct_slice',
    'num_training': 1000,      # Base training data: first 1000 rows
    'num_removes': 30000,      # Update data: next 10000 rows
    'train_mode': 'binary',    # Binary classification mode
}

# Dataset configuration - Dynamic Share
DATASET_CONFIG_DYNAMIC = {
    'dataset': 'dynamic_share',
    'num_training': 5000,      # Base training data: first 1000 rows
    'num_removes': 20000,      # Update data: next 10000 rows
    'train_mode': 'binary',    # Binary classification mode
}

# Dataset configuration - MNIST (classes 3 and 8)
DATASET_CONFIG_MNIST = {
    'dataset': 'mnist',
    'num_training': 1000,      # Base training data: first 1000 rows
    'num_removes': 10000,       # Update data: next 5000 rows
    'train_mode': 'binary',    # Binary classification mode
    # Imbalanced configuration (optional)
    'base_imbalance_ratio': None,  # None means balanced (random), 0.9 means 90% is dominant class
    'incremental_imbalance_ratio': None,  # None means balanced, 0.7 means 70% is dominant class
    'dominant_class': 3,       # Dominant class: 3 or 8
}

# Dataset configuration - Covertype (Forest Cover Type)
DATASET_CONFIG_COVERTYPE = {
    'dataset': 'covertype',
    'num_training': 1000,      # Base training data: first 2000 rows
    'num_removes': 20000,      # Update data: next 50000 rows
    'train_mode': 'binary',    # Binary classification mode
    'binary_task': 'class2_vs_rest',  # 'class1_vs_rest' or 'class2_vs_rest' (default: class2)
}

# Dataset configuration - a1a (Adult Dataset from LIBSVM)
DATASET_CONFIG_A1A = {
    'dataset': 'a1a',
    'num_training': 1000,      # Base training data
    'num_removes': 10000,      # Update data
    'train_mode': 'binary',    # Binary classification mode
}

# Dataset configuration - w8a (from LIBSVM)
DATASET_CONFIG_W8A = {
    'dataset': 'w8a',
    'num_training': 1000,      # Base training data
    'num_removes': 20000,      # Update data
    'train_mode': 'binary',    # Binary classification mode
}

# Common training parameters
COMMON_ARGS = {
    'lam': 1e-2,               # L2 regularization
    'std': 10,                 # Standard deviation for objective perturbation
    'num_steps': 500,          # Number of optimization steps
    'subsample_ratio': 1.0,    # No negative example subsampling
    'train_sep': False,
    'verbose': False,
    'lip_constant': 1/4,       # Lipschitz constant for gradient norm bound
    'gradnorm_constant': 1,    # Gradient norm constant for worst-case bound
}
# MNIST lam,std = (1e-3,10)
# CT_Slice lam,std = (1e-5,20)
# Covertype lam,std = (1e-2,80)
# A1A lam,std = (1e-3,10)
# W8A lam,std = (1e-2,10)
# Dynamic Share lam,std = (1e-2,10)


#-------------------------------------------------------------

# Gradient Norm Experiment Parameters (separate from Pareto experiments)
GRADIENT_NORM_ARGS = {
    'lam': 1e-4,               # L2 regularization
    'std': 0.001,                 # Standard deviation for objective perturbation
    'num_steps': 100,          # Number of optimization steps
    'lip_constant': 1/4,       # Lipschitz constant for gradient norm bound
    'gradnorm_constant': 1,    # Gradient norm constant for worst-case bound
    'weight_constant': 1,      # Weight constant for sampling methods bound (term2)
    'hessian_constant': 1/4,   # Hessian approximation constant for sampling methods (term3)
    'epsilon_uniform': 10,    # Approximation error for Hessian (uniform sampling)
    'epsilon_leverage': 0.1,   # Approximation error for Hessian (leverage sampling)
}

# Uniform sampling probability for gradient norm experiments
GRADIENT_NORM_UNIFORM_PROB = 0.6  # Fixed sampling probability (50%)

# Leverage score sampling parameters for gradient norm experiments
# Single configuration: (beta, lambda_reg)
GRADIENT_NORM_LEVERAGE_PARAMS = (0.9, 0.01) 

# Dataset configuration for Gradient Norm Experiments - CT Slice
GRADIENT_NORM_DATASET_CONFIG = {
    'dataset': 'ct_slice',
    'num_training': 1000,      # Base training data
    'num_removes': 5000,       # Incremental updates for gradient tracking (smaller for detailed analysis)
    'train_mode': 'binary',
}

# Dataset configuration for Gradient Norm Experiments - Dynamic Share
GRADIENT_NORM_DATASET_CONFIG_DYNAMIC = {
    'dataset': 'dynamic_share',
    'num_training': 10000,
    'num_removes': 10000,
    'train_mode': 'binary',
}

# Dataset configuration for Gradient Norm Experiments - MNIST
GRADIENT_NORM_DATASET_CONFIG_MNIST = {
    'dataset': 'mnist',
    'num_training': 1000,      # Base training data
    'num_removes': 5000,       # Incremental updates (total: 1000 + 5000 = 6000)
                               # Note: MNIST has ~12,000 samples (classes 3 & 8)
                               # You can increase these values up to ~11,000 total
    'train_mode': 'binary',
    'base_imbalance_ratio': None,
    'incremental_imbalance_ratio': None,
    'dominant_class': 3,
}

# Dataset configuration for Gradient Norm Experiments - Covertype
GRADIENT_NORM_DATASET_CONFIG_COVERTYPE = {
    'dataset': 'covertype',
    'num_training': 1000,
    'num_removes': 5000,
    'train_mode': 'binary',
    'binary_task': 'class2_vs_rest',
}

# Dataset configuration for Gradient Norm Experiments - a1a
GRADIENT_NORM_DATASET_CONFIG_A1A = {
    'dataset': 'a1a',
    'num_training': 1000,
    'num_removes': 5000,
    'train_mode': 'binary',
}

# Dataset configuration for Gradient Norm Experiments - w8a
GRADIENT_NORM_DATASET_CONFIG_W8A = {
    'dataset': 'w8a',
    'num_training': 1000,
    'num_removes': 5000,
    'train_mode': 'binary',
}

# Streaming (leverage sampling) configurations
# (β, λ) pairs from CT Slice recommendations in OPTIMAL_LEVERAGE_CONFIGURATIONS.md
STREAMING_CONFIGS = [
    # Configuration group: Epsilon-based parameters
    # lambda = 0.01/eps beta = 1/2eps^2
    # eps = 1.0: β = 0.5, λ = 0.01
    (1, 0.01),
    # eps = 0.7: β = 1.02, λ = 0.014
    (1.02, 0.014),
    # eps = 0.5: β = 2.0, λ = 0.02  
    (2.0, 0.02),
    # eps = 0.4: β = 3.13, λ = 0.025
    (3.13, 0.025),
    # eps = 0.3: β = 5.56, λ = 0.033
    (5.56, 0.033),
    # eps = 0.25: β = 8.0, λ = 0.04
    (8.0, 0.04),
    # eps = 0.2: β = 12.5, λ = 0.05
    (12.5, 0.05),
    # eps = 0.15: β = 22.22, λ = 0.067
    (22.22, 0.067),
    # # eps = 0.1: β = 50.0, λ = 0.1
    # (50.0, 0.1),
    # # eps = 0.05: β = 200.0, λ = 0.2
    # (200.0, 0.2),
]

# Uniform sampling configurations
# Different uniform sampling probabilities
UNIFORM_CONFIGS = [
    0.10,   # 10% sampling
    0.15,   # 15% sampling
    0.20,   # 20% sampling
    0.25,   # 25% sampling
    0.30,   # 30% sampling
    0.40,   # 40% sampling
    0.45,   # 45% sampling
    0.70,   # 70% sampling
]

# Streaming configurations for Dynamic Share dataset
# (β, λ) pairs optimized for Dynamic Share (from test_sampling.py experiments)
STREAMING_CONFIGS_DYNAMIC = [
    # Configuration group: Epsilon-based parameters
    # lambda = 0.01/eps beta = 1/2eps^2
    # eps = 1.5: β = 0.222, λ = 0.00667
    (0.222, 0.00667),
    # eps = 1.0: β = 0.5, λ = 0.01
    (1, 0.01),
    # eps = 0.5: β = 2.0, λ = 0.02  
    (2.0, 0.02),
    # eps = 0.4: β = 3.13, λ = 0.025
    (3.13, 0.025),
    # eps = 0.3: β = 5.56, λ = 0.033
    (5.56, 0.033),
    # eps = 0.25: β = 8.0, λ = 0.04
    (8.0, 0.04),
    # eps = 0.2: β = 12.5, λ = 0.05
    (12.5, 0.05),
    
]

# Uniform sampling configurations for Dynamic Share
UNIFORM_CONFIGS_DYNAMIC = [
    0.20,   # 20% sampling
    0.25,   # 25% sampling
    0.30,   # 30% sampling
    0.35,   # 35% sampling
    0.40,   # 40% sampling
    0.50,   # 50% sampling
    0.70,   # 70% sampling
]

# Streaming configurations for MNIST dataset (classes 3 and 8)
STREAMING_CONFIGS_MNIST = [
    # Configuration group: Epsilon-based parameters
    # lambda = 0.01/eps beta = 1/2eps^2
    # eps = 1.5: β = 0.222, λ = 0.00667
    (0.222, 0.00667),
    # eps = 1.0: β = 0.5, λ = 0.01
    (1, 0.01),
    # eps = 0.7: β = 1.02, λ = 0.014
    (1.02, 0.014),
    # eps = 0.5: β = 2.0, λ = 0.02  
    (2.0, 0.02),
    # eps = 0.4: β = 3.13, λ = 0.025
    (3.13, 0.025),
    # eps = 0.3: β = 5.56, λ = 0.033
    (5.56, 0.033),
    # eps = 0.25: β = 8.0, λ = 0.04
    (8.0, 0.04),
    # eps = 0.2: β = 12.5, λ = 0.05
    (12.5, 0.05),
]

# Uniform sampling configurations for MNIST
UNIFORM_CONFIGS_MNIST = [
    0.15,   # 15% sampling
    0.20,   # 20% sampling
    0.25,   # 25% sampling
    0.30,   # 30% sampling
    0.40,   # 40% sampling
    0.50,   # 50% sampling
    0.60,   # 60% sampling
    0.70,   # 70% sampling
]

# Streaming configurations for Covertype dataset
STREAMING_CONFIGS_COVERTYPE = [
    # Configuration group: Epsilon-based parameters
    # lambda = 0.01/eps beta = 1/2eps^2
    # eps = 0.25: β = 8.0, λ = 0.04
    (8.0, 0.04),
    # eps = 0.2: β = 12.5, λ = 0.05
    (12.5, 0.05),
    # eps = 0.15: β = 22.22, λ = 0.067
    (22.22, 0.067),
    # eps = 0.14: β = 25.51, λ = 0.071
    (25.51, 0.071),
    # eps = 0.13: β = 29.59, λ = 0.077
    (29.59, 0.077),
    # eps = 0.125: β = 32.0, λ = 0.08
    (32.0, 0.08),
    # eps = 0.115: β = 37.77, λ = 0.087
    (37.77, 0.087),
    # eps = 0.11: β = 41.32, λ = 0.091
    (41.32, 0.091),
    # eps = 0.1: β = 50.0, λ = 0.1
    (50.0, 0.1),
    # eps = 0.075: β = 88.89, λ = 0.133
    (88.89, 0.133),
]

# Uniform sampling configurations for Covertype
UNIFORM_CONFIGS_COVERTYPE = [
    0.35,   # 35% sampling
    0.40,   # 40% sampling
    0.45,   # 45% sampling
    0.50,   # 50% sampling
    0.55,   # 55% sampling
    0.60,   # 60% sampling
    0.65,   # 65% sampling
    0.70,   # 70% sampling
]

# Streaming configurations for a1a dataset
STREAMING_CONFIGS_A1A = [
    # Configuration group: Epsilon-based parameters
    (2.47, 0.022),
    # eps = 0.4: β = 3.13, λ = 0.025
    (3.13, 0.025),
    # # eps = 0.35: β = 4.08, λ = 0.029
    # (4.08, 0.029),
    # eps = 0.3: β = 5.56, λ = 0.033
    (5.56, 0.033),
    # eps = 0.275: β = 6.61, λ = 0.036
    (6.61, 0.036),
    # # eps = 0.25: β = 8.0, λ = 0.04
    # (8.0, 0.04),
    # # eps = 0.225: β = 9.88, λ = 0.044
    # (9.88, 0.044),
    # # eps = 0.2: β = 12.5, λ = 0.05
    # (12.5, 0.05),
    # eps = 0.175: β = 16.33, λ = 0.057
    (16.33, 0.057),
    # eps = 0.15: β = 22.22, λ = 0.067
    (22.22, 0.067),
    # # eps = 0.1: β = 50.0, λ = 0.1
    # (50.0, 0.1),
    # # eps = 0.05: β = 200.0, λ = 0.2
    # (200.0, 0.2),
]

# Uniform sampling configurations for a1a
UNIFORM_CONFIGS_A1A = [
    0.15,   # 15% sampling
    0.25,   # 25% sampling
    0.40,   # 40% sampling
    0.9,
]

# Streaming configurations for w8a dataset
STREAMING_CONFIGS_W8A = [
    # Configuration group: Epsilon-based parameters
    # eps = 1.0: β = 0.5, λ = 0.01
    (1, 0.01),
    # eps = 0.7: β = 1.02, λ = 0.014
    (1.02, 0.014),
    # eps = 0.5: β = 2.0, λ = 0.02  
    (2.0, 0.02),
    # eps = 0.4: β = 3.13, λ = 0.025
    (3.13, 0.025),
    # eps = 0.3: β = 5.56, λ = 0.033
    (5.56, 0.033),
    # eps = 0.25: β = 8.0, λ = 0.04
    (8.0, 0.04),
    # eps = 0.2: β = 12.5, λ = 0.05
    (12.5, 0.05),
    # eps = 0.15: β = 22.22, λ = 0.067
    (22.22, 0.067),
    # eps = 0.1: β = 50.0, λ = 0.1
    (50.0, 0.1),
    # # eps = 0.05: β = 200.0, λ = 0.2
    # (200.0, 0.2),
]

# Uniform sampling configurations for w8a
UNIFORM_CONFIGS_W8A = [
    0.05,   # 5% sampling
    0.10,   # 10% sampling
    0.15,   # 15% sampling
    0.20,   # 20% sampling
    0.30,   # 30% sampling
    0.40,   # 40% sampling
    0.50,   # 50% sampling
    0.60,   # 60% sampling
    0.70,   # 70% sampling
]

# Epsilon parameter (deprecated - no longer used as global parameter)
# In streaming: epsilon = 0.01 / lambda_reg (calculated per configuration)
# In uniform: epsilon = 0.5 (not used)
EPSILON = 0.5  # Kept for backward compatibility, but not used

# Output paths (relative to pareto_experiments directory)
OUTPUT_CONFIG = {
    'csv_path': 'results/csv/pareto_comparison_ct_slice.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_ct_slice_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_ct_slice.png',
}

# Output paths for Dynamic Share dataset
OUTPUT_CONFIG_DYNAMIC = {
    'csv_path': 'results/csv/pareto_comparison_dynamic_share.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_dynamic_share_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_dynamic_share.png',
}

# Output paths for MNIST dataset
OUTPUT_CONFIG_MNIST = {
    'csv_path': 'results/csv/pareto_comparison_mnist.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_mnist_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_mnist.png',
}

# Output paths for Covertype dataset
OUTPUT_CONFIG_COVERTYPE = {
    'csv_path': 'results/csv/pareto_comparison_covertype.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_covertype_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_covertype.png',
}

# Output paths for a1a dataset
OUTPUT_CONFIG_A1A = {
    'csv_path': 'results/csv/pareto_comparison_a1a.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_a1a_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_a1a.png',
}

# Output paths for w8a dataset
OUTPUT_CONFIG_W8A = {
    'csv_path': 'results/csv/pareto_comparison_w8a.csv',
    'csv_agg_path': 'results/csv/pareto_comparison_w8a_aggregated.csv',
    'plot_path': 'results/plots/improved/pareto_frontier_w8a.png',
}

#-------------------------------------------------------------
# Incremental Data Ratio Experiment Configurations
# These configurations are used for experiments testing different ratios of incremental data
#-------------------------------------------------------------

# Incremental data ratios to test (as fractions of num_removes)
INCREMENTAL_DATA_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Number of optimization steps for incremental ratio experiments (same for all datasets)
INCREMENTAL_RATIO_NUM_STEPS = 500

# Selected configurations for incremental data ratio experiments
# Each dataset has one streaming (beta, lambda) and one uniform (prob) configuration

# CT Slice - selected configurations for incremental ratio experiments
INCREMENTAL_RATIO_CONFIG_CT_SLICE = {
    'streaming': (12.5, 0.05),      
    'uniform': 0.70,              # 30% uniform sampling
    'lam': 1e-5,                  # L2 regularization (recommended: 1e-5 to 1e-2)
    'std': 20,                    # Objective perturbation std (recommended: 20)
    'train_ratio': 0.7,           # Train/test split ratio
    'temporal_split': False,       # Use temporal split (True for time-series data)
}

# Dynamic Share - selected configurations for incremental ratio experiments
INCREMENTAL_RATIO_CONFIG_DYNAMIC = {
    'streaming': (8, 0.04),  # eps = 0.3: β = 5.56, λ = 0.033
    'uniform': 0.70,              # 40% uniform sampling
    'lam': 1e-2,                  # L2 regularization
    'std': 10,                    # Objective perturbation std
    'train_ratio': 0.9,           # Train/test split ratio
    'temporal_split': True,       # Use temporal split (True for time-series data)
}

# MNIST - selected configurations for incremental ratio experiments
INCREMENTAL_RATIO_CONFIG_MNIST = {
    'streaming': (3.13, 0.025),  # eps = 0.4: β = 3.13, λ = 0.025
    'uniform': 0.60,              # 40% uniform sampling
    'lam': 1e-3,                  # L2 regularization
    'std': 10,                    # Objective perturbation std
    'train_ratio': 0.7,           # Train/test split ratio (MNIST has separate test set)
    'temporal_split': False,      # Use random split (MNIST is not time-series)
}

# W8A - selected configurations for incremental ratio experiments
INCREMENTAL_RATIO_CONFIG_W8A = {
    'streaming': (12.5, 0.05),  # eps = 0.3: β = 5.56, λ = 0.033
    'uniform': 0.70,              # 30% uniform sampling
    'lam': 1e-2,                  # L2 regularization
    'std': 10,                    # Objective perturbation std
    'train_ratio': 0.7,           # Train/test split ratio (W8A has separate test set)
    'temporal_split': False,      # Use random split (W8A is not time-series)
}

# Output configuration for incremental data ratio experiments
OUTPUT_CONFIG_INCREMENTAL_RATIO = {
    'csv_path': 'results/csv/incremental_ratio_comparison.csv',
    'csv_agg_path': 'results/csv/incremental_ratio_comparison_aggregated.csv',
    'plot_path': 'results/plots/incremental_ratio_comparison.png',
}

