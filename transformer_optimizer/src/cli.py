"""
Command-line interface for transformer optimization.
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import OptimizationConfig
from .optimizer import TransformerOptimizer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="End-to-End Transformer Model Optimization with ONNX Runtime and Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with default configuration
  python -m transformer_optimizer.src.cli
  
  # Optimize with custom model
  python -m transformer_optimizer.src.cli --model-id bert-base-uncased
  
  # Use configuration file
  python -m transformer_optimizer.src.cli --config config.json
  
  # Disable quantization
  python -m transformer_optimizer.src.cli --no-quantization
  
  # Run on GPU
  python -m transformer_optimizer.src.cli --device cuda
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-id",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Hugging Face model ID"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for inference"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for models and results"
    )
    
    # Optimization options
    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Skip ONNX export"
    )
    
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Skip quantization"
    )
    
    parser.add_argument(
        "--quantization-approach",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization approach"
    )
    
    parser.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Disable torch.compile optimization"
    )
    
    # Benchmark options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=8,
        help="Number of benchmark iterations"
    )
    
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip benchmarking"
    )
    
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Skip accuracy evaluation"
    )
    
    # Dataset options
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="glue",
        help="Dataset name"
    )
    
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="sst2",
        help="Dataset configuration"
    )
    
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="validation[:20%]",
        help="Dataset split to use"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to JSON file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = OptimizationConfig.from_json(args.config)
        else:
            # Create config from CLI arguments
            config = OptimizationConfig(
                model_id=args.model_id,
                device=args.device,
                output_dir=Path(args.output_dir),
                onnx_dir=Path(args.output_dir) / "onnx",
                quantized_dir=Path(args.output_dir) / "quantized",
                export_onnx=not args.no_onnx,
                export_quantized=not args.no_quantization,
                quantization_approach=args.quantization_approach,
                use_torch_compile=not args.no_torch_compile,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_benchmark_iterations=args.num_iterations,
                benchmark_enabled=not args.no_benchmark,
                evaluation_enabled=not args.no_evaluation,
                dataset_name=args.dataset_name,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                log_level=args.log_level,
            )
        
        # Save configuration if requested
        if args.save_config:
            config.save(args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")
        
        # Run optimization
        optimizer = TransformerOptimizer(config)
        summary = optimizer.optimize()
        
        # Print results
        optimizer.print_results_table()
        
        # Print speedup analysis
        speedups = optimizer.get_speedup_analysis()
        if speedups:
            print("\nSpeedup Analysis (vs PyTorch Eager):")
            print("-" * 40)
            for model_name, speedup in speedups.items():
                print(f"{model_name}: {speedup:.2f}x")
        
        print("\nOptimization complete! Results saved to:", config.output_dir)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
