README.md
## 🚀 Quick Start

### Step 1: Verify Environment

```bash
python verify_setup.py
```

### Step 2: Run Experiments

#### Pareto Frontier Experiments

```bash
# Use the quick run script (recommended)
chmod +x run_experiments.sh

# CT Slice - Random Split
./run_experiments.sh ct-slice-random

# Dynamic Share - Random Split
./run_experiments.sh dynamic-random

# MNIST - Random Split
./run_experiments.sh mnist-random

# Quick Test (Single Run)
./run_experiments.sh test-ct
```

or directly with Python:

```bash
# CT Slice dataset (5 repetitions)
python compare_methods_v2.py --dataset ct_slice --num-repeats 5

# Dynamic Share dataset (5 repetitions)
python compare_methods_v2.py --dataset dynamic_share --num-repeats 5

# MNIST dataset (5 repetitions)
python compare_methods_v2.py --dataset mnist --num-repeats 5
```

#### Gradient Norm Tracking Experiments

```bash
# Using Shell Script
./run_experiments.sh gradnorm-ct       # CT Slice
./run_experiments.sh gradnorm-dynamic  # Dynamic Share
./run_experiments.sh gradnorm-mnist    # MNIST
./run_experiments.sh gradnorm-all      # All datasets

# Or directly with Python
python run_gradient_norm_experiment.py --dataset ct_slice --data-dir ../data
python run_gradient_norm_experiment.py --dataset dynamic_share --data-dir ../data
python run_gradient_norm_experiment.py --dataset mnist --data-dir ../data
```

### Step 3: View Results

#### Pareto Frontier Experiment Results

```bash
# CSV result files
ls results/csv/

# Pareto frontier plots
ls results/plots/

# View aggregated results (with mean and standard deviation)
cat results/csv/pareto_comparison_ct_slice_aggregated.csv
```

#### Gradient Norm Tracking Experiment Results

```bash
# CSV results
ls results/csv/gradient_norm/

# Comparison plots
ls results/plots/gradient_norm/

# View summary
cat results/csv/gradient_norm/gradient_norm_mnist_summary.csv
```

## 📖 Experimental Methods

This project compares three incremental learning methods:

### 1. **Incremental (Full Updates)** 
Full incremental update baseline
- Uses all arriving data points
- Highest accuracy but slowest
- Serves as performance upper bound

### 2. **Streaming (Leverage Sampling)**
Leverage score-based sampling
- Intelligently selects important data points
- Adaptive sampling probability
- Theoretically guaranteed approximation quality

### 3. **Uniform Sampling**
Uniform random sampling
- Fixed probability sampling
- Simple but effective
- Serves as sampling baseline

## 📈 Result Analysis

### Pareto Frontier Plot Interpretation

Generated charts show:
- **X-axis**: Total update time (seconds) - smaller is better
- **Y-axis**: Test accuracy - larger is better
- **Ideal Region**: Top-left corner (fast and accurate)

Three methods with different markers:
- 🔴 **Red stars**: Incremental (full update baseline)
- 🔵 **Blue circles**: Streaming (Leverage sampling)
- 🟢 **Green squares**: Uniform (uniform sampling)

### Output File Structure

```
results/
├── csv/
│   ├── pareto_comparison_ct_slice.csv                    # CT Slice raw results
│   ├── pareto_comparison_ct_slice_aggregated.csv         # CT Slice aggregated results
│   ├── pareto_comparison_dynamic_share.csv               # Dynamic Share raw results
│   ├── pareto_comparison_dynamic_share_aggregated.csv    # Dynamic Share aggregated results
│   ├── pareto_comparison_mnist.csv                       # MNIST raw results
│   ├── pareto_comparison_mnist_aggregated.csv            # MNIST aggregated results
│   └── gradient_norm/                                    # Gradient norm experiment results
│       ├── gradient_norm_ct_slice_incremental.csv        # Detailed results (each iteration)
│       ├── gradient_norm_ct_slice_uniform.csv
│       ├── gradient_norm_ct_slice_streaming.csv
│       ├── gradient_norm_ct_slice_summary.csv            # Summary (final statistics)
│       ├── gradient_norm_dynamic_share_*.csv             # Dynamic Share results
│       └── gradient_norm_mnist_*.csv                     # MNIST results
└── plots/
    ├── pareto_frontier_ct_slice.png                      # CT Slice Pareto frontier
    ├── pareto_frontier_dynamic_share.png                 # Dynamic Share Pareto frontier
    ├── pareto_frontier_mnist.png                         # MNIST Pareto frontier
    └── gradient_norm/                                    # Gradient norm comparison plots
        ├── gradient_norm_comparison_ct_slice.png         # CT Slice gradient norm
        ├── gradient_norm_comparison_dynamic_share.png    # Dynamic Share gradient norm
        └── gradient_norm_comparison_mnist.png            # MNIST gradient norm
```

## ⚙️ Command Line Arguments

```bash
python compare_methods_v2.py \
    --dataset {ct_slice|dynamic_share|mnist}  # Dataset selection
    --data-dir PATH                            # Data directory (default: ../data)
    --num-repeats N                            # Number of repetitions (default: 5)
    --train-ratio RATIO                        # Training ratio (default: 0.7)
    --temporal-split                           # Time series split (default: random)
    --skip-incremental                         # Skip Incremental
    --skip-streaming                           # Skip Streaming
    --skip-uniform                             # Skip Uniform
```

## 🎯 Recommended Workflow

### First-time Use
```bash
# 1. Verify environment
python verify_setup.py

# 2. Quick test
./run_experiments.sh test-ct

# 3. View results
ls results/
```

### Formal Experiments
```bash
# 1. Run complete experiment (5 repetitions)
./run_experiments.sh ct-slice-random

# 2. View aggregated results
cat results/csv/pareto_comparison_ct_slice_aggregated.csv

# 3. View charts
open results/plots/pareto_frontier_ct_slice.png
```

### Multi-dataset Comparison
```bash
# Run all datasets
./run_experiments.sh ct-slice-random
./run_experiments.sh dynamic-random
./run_experiments.sh mnist-random

# Compare results
cat results/csv/pareto_comparison_*_aggregated.csv
```

### 🔬 Experiment Examples

### Example 1: CT Slice Basic Experiment
```bash
python compare_methods_v2.py \
    --dataset ct_slice \
    --num-repeats 5
```

### Example 2: Dynamic Share Time Series Split
```bash
python compare_methods_v2.py \
    --dataset dynamic_share \
    --temporal-split \
    --num-repeats 5
```

### Example 3: Test Streaming Method Only
```bash
python compare_methods_v2.py \
    --dataset mnist \
    --skip-incremental \
    --skip-uniform \
    --num-repeats 3
```