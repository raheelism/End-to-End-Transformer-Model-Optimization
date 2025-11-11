"""
Benchmarking and evaluation utilities for transformer models.
"""

import logging
import time
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
import evaluate
from transformers import PreTrainedTokenizer

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class Benchmarker:
    """Handles benchmarking and evaluation of models."""
    
    def __init__(self, config: OptimizationConfig, tokenizer: PreTrainedTokenizer):
        """
        Initialize Benchmarker.
        
        Args:
            config: Optimization configuration
            tokenizer: Tokenizer for the model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset: Optional[Dataset] = None
        self.texts: Optional[List[str]] = None
        self.labels: Optional[List[int]] = None
        self.metric = None
        
        logger.info("Benchmarker initialized")
    
    def load_dataset(self) -> Tuple[List[str], List[int]]:
        """
        Load evaluation dataset.
        
        Returns:
            Tuple of (texts, labels)
        """
        if self.texts is not None and self.labels is not None:
            return self.texts, self.labels
        
        logger.info(f"Loading dataset: {self.config.dataset_name}/{self.config.dataset_config}")
        
        try:
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                split=self.config.dataset_split
            )
            
            self.texts = self.dataset["sentence"]
            self.labels = self.dataset["label"]
            
            logger.info(f"Dataset loaded: {len(self.texts)} samples")
            
            return self.texts, self.labels
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_metric(self):
        """Load evaluation metric."""
        if self.metric is None:
            logger.info(f"Loading metric: {self.config.metric_name}")
            self.metric = evaluate.load(self.config.metric_name)
        return self.metric
    
    def make_batches(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        Create batches of tokenized texts.
        
        Args:
            texts: List of text strings
            
        Yields:
            Batched tokenized inputs
        """
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            yield self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
    
    def run_evaluation(
        self,
        predict_fn: Callable,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Args:
            predict_fn: Function that takes tokenized inputs and returns predictions
            texts: Texts to evaluate on. If None, uses loaded dataset
            labels: True labels. If None, uses loaded dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if texts is None or labels is None:
            texts, labels = self.load_dataset()
        
        metric = self.load_metric()
        
        logger.info("Running evaluation...")
        predictions = []
        
        try:
            for batch_tokens in self.make_batches(texts):
                batch_preds = predict_fn(batch_tokens)
                predictions.extend(batch_preds)
            
            results = metric.compute(predictions=predictions, references=labels)
            logger.info(f"Evaluation complete: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def benchmark(
        self,
        predict_fn: Callable,
        texts: Optional[List[str]] = None,
        num_warmup: Optional[int] = None,
        num_iterations: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            predict_fn: Function that takes tokenized inputs and returns predictions
            texts: Texts to benchmark on. If None, uses loaded dataset
            num_warmup: Number of warmup runs. If None, uses config value
            num_iterations: Number of benchmark iterations. If None, uses config value
            
        Returns:
            Dictionary with timing statistics (mean, std, min, max in milliseconds)
        """
        if texts is None:
            texts, _ = self.load_dataset()
        
        if num_warmup is None:
            num_warmup = self.config.num_warmup_runs
        
        if num_iterations is None:
            num_iterations = self.config.num_benchmark_iterations
        
        logger.info(f"Running benchmark with {num_warmup} warmup runs and {num_iterations} iterations")
        
        # Warmup runs
        warmup_texts = texts[:self.config.batch_size * 2]
        for i in range(num_warmup):
            for batch_tokens in self.make_batches(warmup_texts):
                predict_fn(batch_tokens)
        
        # Benchmark runs
        times = []
        for i in range(num_iterations):
            start_time = time.time()
            for batch_tokens in self.make_batches(texts):
                predict_fn(batch_tokens)
            elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
            times.append(elapsed)
        
        results = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
        
        logger.info(f"Benchmark complete: {results['mean_ms']:.1f}±{results['std_ms']:.1f} ms")
        
        return results
    
    def run_full_benchmark(
        self,
        predict_fn: Callable,
        model_name: str,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Run both benchmarking and evaluation.
        
        Args:
            predict_fn: Function that takes tokenized inputs and returns predictions
            model_name: Name of the model being benchmarked
            texts: Texts to use. If None, uses loaded dataset
            labels: Labels to use. If None, uses loaded dataset
            
        Returns:
            Dictionary with all results
        """
        logger.info(f"Running full benchmark for {model_name}")
        
        results = {"model_name": model_name}
        
        # Run benchmarking
        if self.config.benchmark_enabled:
            timing_results = self.benchmark(predict_fn, texts=texts)
            results.update(timing_results)
        
        # Run evaluation
        if self.config.evaluation_enabled:
            eval_results = self.run_evaluation(predict_fn, texts=texts, labels=labels)
            results.update(eval_results)
        
        return results
