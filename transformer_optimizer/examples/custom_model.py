"""
Example of optimizing a custom model with specific configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import OptimizationConfig
from optimizer import TransformerOptimizer


def main():
    """Run custom model optimization example."""
    print("=" * 80)
    print("Transformer Optimization - Custom Model Example")
    print("=" * 80)
    
    # Create configuration for a different model
    config = OptimizationConfig(
        model_id="bert-base-uncased",  # Different model
        output_dir=Path("outputs/bert_optimization"),
        batch_size=8,  # Smaller batch size for BERT
        max_length=256,  # Longer sequences
        num_benchmark_iterations=3,
        quantization_approach="dynamic",
        use_torch_compile=True,
        log_level="INFO",
        # Custom dataset
        dataset_name="glue",
        dataset_config="mrpc",
        dataset_split="validation[:10%]"
    )
    
    print(f"\nOptimizing model: {config.model_id}")
    print(f"Dataset: {config.dataset_name}/{config.dataset_config}")
    print(f"Output directory: {config.output_dir}")
    
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
