"""
Main optimizer orchestration for transformer model optimization pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

import torch

from .config import OptimizationConfig
from .model_manager import ModelManager
from .quantizer import ModelQuantizer
from .benchmark import Benchmarker

logger = logging.getLogger(__name__)


class TransformerOptimizer:
    """Main class orchestrating the end-to-end optimization pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize TransformerOptimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self._setup_logging()
        self._setup_environment()
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.quantizer = ModelQuantizer(config)
        
        # Will be initialized after tokenizer is loaded
        self.benchmarker: Optional[Benchmarker] = None
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        
        logger.info(f"TransformerOptimizer initialized for model: {config.model_id}")
        logger.info(f"Device: {self.model_manager.device} | PyTorch: {torch.__version__}")
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _setup_environment(self):
        """Set up environment variables."""
        os.environ.setdefault("OMP_NUM_THREADS", self.config.omp_num_threads)
        os.environ.setdefault("MKL_NUM_THREADS", self.config.mkl_num_threads)
        logger.info(f"Environment configured: OMP_NUM_THREADS={self.config.omp_num_threads}, "
                   f"MKL_NUM_THREADS={self.config.mkl_num_threads}")
    
    def _get_pytorch_predict_fn(self, model):
        """Create prediction function for PyTorch model."""
        device = self.model_manager.device
        
        @torch.no_grad()
        def predict(tokens):
            tokens = {k: v.to(device) for k, v in tokens.items()}
            logits = model(**tokens).logits
            return logits.argmax(-1).detach().cpu().tolist()
        
        return predict
    
    def _get_onnx_predict_fn(self, model):
        """Create prediction function for ONNX model."""
        @torch.no_grad()
        def predict(tokens):
            logits = model(**{k: v.cpu() for k, v in tokens.items()}).logits
            return logits.argmax(-1).cpu().tolist()
        
        return predict
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("=" * 60)
        logger.info("Starting optimization pipeline")
        logger.info("=" * 60)
        
        # Create output directories
        self.config.create_output_dirs()
        
        # Load tokenizer
        tokenizer = self.model_manager.load_tokenizer()
        self.benchmarker = Benchmarker(self.config, tokenizer)
        
        # 1. Benchmark PyTorch model
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Benchmarking PyTorch eager model")
        logger.info("=" * 60)
        pytorch_model = self.model_manager.load_pytorch_model()
        pytorch_predict = self._get_pytorch_predict_fn(pytorch_model)
        pytorch_results = self.benchmarker.run_full_benchmark(
            pytorch_predict,
            "PyTorch Eager"
        )
        self.results.append(pytorch_results)
        logger.info(f"PyTorch results: {pytorch_results}")
        
        # 2. Benchmark torch.compile model (if enabled)
        if self.config.use_torch_compile:
            logger.info("\n" + "=" * 60)
            logger.info("Step 2: Benchmarking torch.compile model")
            logger.info("=" * 60)
            compiled_model = self.model_manager.load_compiled_model()
            if compiled_model is not None:
                compiled_predict = self._get_pytorch_predict_fn(compiled_model)
                compiled_results = self.benchmarker.run_full_benchmark(
                    compiled_predict,
                    "torch.compile"
                )
                self.results.append(compiled_results)
                logger.info(f"torch.compile results: {compiled_results}")
        
        # 3. Export to ONNX and benchmark
        if self.config.export_onnx:
            logger.info("\n" + "=" * 60)
            logger.info("Step 3: Exporting to ONNX and benchmarking")
            logger.info("=" * 60)
            self.model_manager.export_to_onnx()
            onnx_model = self.model_manager.load_onnx_model()
            onnx_predict = self._get_onnx_predict_fn(onnx_model)
            onnx_results = self.benchmarker.run_full_benchmark(
                onnx_predict,
                "ONNX Runtime"
            )
            self.results.append(onnx_results)
            logger.info(f"ONNX results: {onnx_results}")
        
        # 4. Quantize ONNX model and benchmark
        if self.config.export_quantized and self.config.export_onnx:
            logger.info("\n" + "=" * 60)
            logger.info("Step 4: Quantizing ONNX model and benchmarking")
            logger.info("=" * 60)
            self.quantizer.quantize_model(self.config.onnx_dir)
            quantized_model = self.quantizer.load_quantized_model(
                device=self.model_manager.device
            )
            quantized_predict = self._get_onnx_predict_fn(quantized_model)
            quantized_results = self.benchmarker.run_full_benchmark(
                quantized_predict,
                "ONNX Quantized"
            )
            self.results.append(quantized_results)
            logger.info(f"Quantized results: {quantized_results}")
        
        # Generate summary
        logger.info("\n" + "=" * 60)
        logger.info("Optimization Complete - Summary")
        logger.info("=" * 60)
        summary = self._generate_summary()
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        return summary
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate optimization summary."""
        summary = {
            "config": self.config.to_dict(),
            "model_info": self.model_manager.get_model_info(),
            "results": self.results,
        }
        
        # Add model size information
        if self.config.export_onnx:
            summary["onnx_model_size"] = self.quantizer.get_model_size(self.config.onnx_dir)
        
        if self.config.export_quantized:
            summary["quantized_model_size"] = self.quantizer.get_model_size(self.config.quantized_dir)
        
        return summary
    
    def _save_results(self):
        """Save results to files."""
        results_dir = self.config.output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        if self.config.results_format == "json" or self.config.results_format == "both":
            json_path = results_dir / "results.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {json_path}")
        
        # Save as CSV
        if self.config.results_format == "csv" or self.config.results_format == "both":
            csv_path = results_dir / "results.csv"
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        # Save summary
        summary = self._generate_summary()
        summary_path = results_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_path}")
    
    def print_results_table(self):
        """Print results in a formatted table."""
        if not self.results:
            logger.warning("No results to display")
            return
        
        df = pd.DataFrame(self.results)
        
        # Reorder columns for better readability
        column_order = ["model_name"]
        if "mean_ms" in df.columns:
            column_order.extend(["mean_ms", "std_ms"])
        if "accuracy" in df.columns:
            column_order.append("accuracy")
        
        # Add any remaining columns
        for col in df.columns:
            if col not in column_order:
                column_order.append(col)
        
        df = df[column_order]
        
        print("\n" + "=" * 80)
        print("Optimization Results")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    def get_speedup_analysis(self) -> Dict[str, float]:
        """
        Calculate speedup compared to baseline PyTorch model.
        
        Returns:
            Dictionary with speedup factors for each model
        """
        if not self.results:
            return {}
        
        baseline = None
        for result in self.results:
            if result["model_name"] == "PyTorch Eager" and "mean_ms" in result:
                baseline = result["mean_ms"]
                break
        
        if baseline is None:
            logger.warning("No PyTorch baseline found for speedup calculation")
            return {}
        
        speedups = {}
        for result in self.results:
            if "mean_ms" in result and result["model_name"] != "PyTorch Eager":
                speedup = baseline / result["mean_ms"]
                speedups[result["model_name"]] = speedup
        
        return speedups
