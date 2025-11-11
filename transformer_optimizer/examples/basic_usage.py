"""
Basic usage example for transformer optimization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import OptimizationConfig
from optimizer import TransformerOptimizer


def main():
    """Run basic optimization example."""
    print("=" * 80)
    print("Transformer Optimization - Basic Example")
    print("=" * 80)
    
    # Create configuration
    config = OptimizationConfig(
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        output_dir=Path("outputs/basic_example"),
        batch_size=16,
        num_benchmark_iterations=5,
        log_level="INFO"
    )
    
    # Create optimizer
    optimizer = TransformerOptimizer(config)
    
    # Run optimization
    summary = optimizer.optimize()
    
    # Display results
    optimizer.print_results_table()
    
    # Show speedup analysis
    speedups = optimizer.get_speedup_analysis()
    if speedups:
        print("\n" + "=" * 80)
        print("Speedup Analysis (compared to PyTorch Eager)")
        print("=" * 80)
        for model_name, speedup in speedups.items():
            print(f"  {model_name:<30} {speedup:.2f}x faster")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
