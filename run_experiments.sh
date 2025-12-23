#!/bin/bash

# Pareto Experiments V3.0 - Unified Run Script
# Pareto Experiments V3.0 - Unified Run Script

echo "=========================================="
echo "Pareto Experiments V3.0"
echo "Incremental Learning Pareto Frontier Comparison"
echo "=========================================="
echo ""

# Check if parameters are passed
if [ "$#" -eq 0 ]; then
    echo "Usage | Usage:"
    echo "  ./run_experiments.sh [options]"
    echo ""
    echo "Dataset Options | Dataset Options:"
    echo "  ct-slice-random       - CT Slice dataset, random split"
    echo "  ct-slice-temporal     - CT Slice dataset, temporal split"
    echo "  dynamic-random        - Dynamic Share dataset, random split"
    echo "  dynamic-temporal      - Dynamic Share dataset, temporal split"
    echo "  mnist-random          - MNIST dataset (3 and 8), random split"
    echo "  mnist-temporal        - MNIST dataset (3 and 8), temporal split"
    echo ""
    echo "Quick Tests | Quick Tests:"
    echo "  test-ct               - Quick test CT Slice (single run, ~30 minutes)"
    echo "  test-dynamic          - Quick test Dynamic Share (single run, ~40 minutes)"
    echo "  test-mnist            - Quick test MNIST (single run, ~25 minutes)"
    echo ""
    echo "Gradient Norm Tracking | Gradient Norm Tracking:"
    echo "  gradnorm-ct           - CT Slice gradient norm experiment (~15 minutes)"
    echo "  gradnorm-dynamic      - Dynamic Share gradient norm experiment (~25 minutes)"
    echo "  gradnorm-mnist        - MNIST gradient norm experiment (~10 minutes)"
    echo "  gradnorm-all          - All datasets gradient norm experiment (~50 minutes)"
    echo ""
    echo "Batch Runs | Batch Runs:"
    echo "  all                   - Run all experiments (6 full experiments, ~16-20 hours)"
    echo "  all-random            - Run all datasets with random splits (3 experiments, ~12-15 hours)"
    echo "  all-temporal          - Run all datasets with temporal splits (3 experiments, ~12-15 hours)"
    echo "  quick-all             - Quick test all datasets (3 tests, ~1.5 hours)"
    echo ""
    echo "Examples | Examples:"
    echo "  ./run_experiments.sh test-ct              # Quick test"
    echo "  ./run_experiments.sh ct-slice-random      # Full CT Slice experiment"
    echo "  ./run_experiments.sh all-random           # All datasets random splits"
    echo ""
    exit 1
fi

# Default parameters
NUM_REPEATS=5
DATA_DIR="../data"

# Run experiments based on options
case "$1" in
    # ==================== CT Slice ====================
    ct-slice-random)
        echo "🔬 Running experiment: CT Slice dataset - Random split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 4-6 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    ct-slice-temporal)
        echo "🔬 Running experiment: CT Slice dataset - Temporal split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 4-6 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    # ==================== Dynamic Share ====================
    dynamic-random)
        echo "📈 Running experiment: Dynamic Share dataset - Random split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 6-8 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    dynamic-temporal)
        echo "📈 Running experiment: Dynamic Share dataset - Temporal split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 6-8 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    # ==================== MNIST ====================
    mnist-random)
        echo "🔢 Running experiment: MNIST dataset (3 and 8) - Random split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 3-5 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    mnist-temporal)
        echo "🔢 Running experiment: MNIST dataset (3 and 8) - Temporal split"
        echo "   Number of repeats: ${NUM_REPEATS}"
        echo "   Estimated time: 3-5 hours"
        echo ""
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        ;;
    
    # ==================== Quick Tests ====================
    test-ct)
        echo "⚡ Quick test: CT Slice dataset (single run)"
        echo "   Estimated time: 30 minutes"
        echo ""
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        ;;
    
    test-dynamic)
        echo "⚡ Quick test: Dynamic Share dataset (single run)"
        echo "   Estimated time: 40 minutes"
        echo ""
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        ;;
    
    test-mnist)
        echo "⚡ Quick test: MNIST dataset (single run)"
        echo "   Estimated time: 25 minutes"
        echo ""
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        ;;
    
    # ==================== Gradient Norm Tracking Experiments ====================
    gradnorm-ct)
        echo "📊 Gradient norm tracking experiment: CT Slice dataset"
        echo "   Estimated time: 15 minutes"
        echo "   Methods: Incremental + Uniform + Streaming"
        echo ""
        python run_gradient_norm_experiment.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR}
        ;;
    
    gradnorm-dynamic)
        echo "📊 Gradient norm tracking experiment: Dynamic Share dataset"
        echo "   Estimated time: 25 minutes"
        echo "   Methods: Incremental + Uniform + Streaming"
        echo ""
        python run_gradient_norm_experiment.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR}
        ;;
    
    gradnorm-mnist)
        echo "📊 Gradient norm tracking experiment: MNIST dataset"
        echo "   Estimated time: 10 minutes"
        echo "   Methods: Incremental + Uniform + Streaming"
        echo ""
        python run_gradient_norm_experiment.py \
            --dataset mnist \
            --data-dir ${DATA_DIR}
        ;;
    
    gradnorm-all)
        echo "📊 Gradient norm tracking experiment: All datasets"
        echo "   Estimated time: 50 minutes"
        echo ""
        
        echo "=== 1/3: CT Slice ==="
        python run_gradient_norm_experiment.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR}
        
        echo ""
        echo "=== 2/3: Dynamic Share ==="
        python run_gradient_norm_experiment.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR}
        
        echo ""
        echo "=== 3/3: MNIST ==="
        python run_gradient_norm_experiment.py \
            --dataset mnist \
            --data-dir ${DATA_DIR}
        
        echo ""
        echo "🎉 All gradient norm experiments completed!"
        ;;
    
    # ==================== Batch Runs ====================
    all)
        echo "🚀 Running all experiments (6 full experiments)"
        echo "   ⚠️  Warning: This will take 16-20 hours"
        echo "   Recommendation: Run in background or use screen/tmux"
        echo ""
        read -p "Confirm to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled"
            exit 1
        fi
        
        echo ""
        echo "=== 1/6: CT Slice - Random split ==="
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 2/6: CT Slice - Temporal split ==="
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 3/6: Dynamic Share - Random split ==="
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 4/6: Dynamic Share - Temporal split ==="
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 5/6: MNIST - Random split ==="
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 6/6: MNIST - Temporal split ==="
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "🎉 All experiments completed!"
        ;;
    
    all-random)
        echo "🚀 Running all datasets - Random splits (3 experiments)"
        echo "   Estimated time: 12-15 hours"
        echo ""
        
        echo "=== 1/3: CT Slice ==="
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 2/3: Dynamic Share ==="
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 3/3: MNIST ==="
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "🎉 All random split experiments completed!"
        ;;
    
    all-temporal)
        echo "🚀 Running all datasets - Temporal splits (3 experiments)"
        echo "   Estimated time: 12-15 hours"
        echo ""
        
        echo "=== 1/3: CT Slice ==="
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 2/3: Dynamic Share ==="
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "=== 3/3: MNIST ==="
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --temporal-split \
            --num-repeats ${NUM_REPEATS}
        
        echo ""
        echo "🎉 All temporal split experiments completed!"
        ;;
    
    quick-all)
        echo "⚡ Quick test all datasets (single run)"
        echo "   Estimated time: 1.5 hours"
        echo ""
        
        echo "=== 1/3: CT Slice ==="
        python compare_methods_v2.py \
            --dataset ct_slice \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        
        echo ""
        echo "=== 2/3: Dynamic Share ==="
        python compare_methods_v2.py \
            --dataset dynamic_share \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        
        echo ""
        echo "=== 3/3: MNIST ==="
        python compare_methods_v2.py \
            --dataset mnist \
            --data-dir ${DATA_DIR} \
            --num-repeats 1
        
        echo ""
        echo "🎉 All quick tests completed!"
        ;;
    
    *)
        echo "❌ Error: Unknown option '$1'"
        echo "   Run './run_experiments.sh' to see available options"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Experiments completed!"
echo "=========================================="
echo "📁 Results files are located at:"
echo "   - CSV files: results/csv/"
echo "   - Plot files: results/plots/"
echo ""
echo "📊 View results:"
echo "   ls results/csv/"
echo "   ls results/plots/"
echo "   cat results/csv/pareto_comparison_*_aggregated.csv"
echo ""
echo "📚 More information:"
echo "   cat README.md"
echo "   cat QUICKSTART.md"
echo ""
