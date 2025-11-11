"""
Model loading and management for transformer optimization.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from optimum.onnxruntime import ORTModelForSequenceClassification

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, conversion, and caching."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize ModelManager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.device = self._get_device()
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.pytorch_model: Optional[PreTrainedModel] = None
        self.onnx_model: Optional[ORTModelForSequenceClassification] = None
        self.compiled_model: Optional[PreTrainedModel] = None
        
        logger.info(f"ModelManager initialized with device: {self.device}")
    
    def _get_device(self) -> str:
        """Determine the device to use for computation."""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        
        return device
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and cache the tokenizer."""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.config.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            logger.info("Tokenizer loaded successfully")
        return self.tokenizer
    
    def load_pytorch_model(self) -> PreTrainedModel:
        """Load and cache the PyTorch model."""
        if self.pytorch_model is None:
            logger.info(f"Loading PyTorch model: {self.config.model_id}")
            self.pytorch_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_id
            ).to(self.device).eval()
            logger.info(f"PyTorch model loaded successfully on {self.device}")
        return self.pytorch_model
    
    def load_compiled_model(self) -> Optional[PreTrainedModel]:
        """Load and cache the torch.compile optimized model."""
        if not self.config.use_torch_compile:
            logger.info("torch.compile disabled in configuration")
            return None
        
        if self.compiled_model is None:
            try:
                pytorch_model = self.load_pytorch_model()
                logger.info("Compiling model with torch.compile...")
                self.compiled_model = torch.compile(
                    pytorch_model,
                    mode=self.config.torch_compile_mode,
                    fullgraph=self.config.torch_compile_fullgraph
                )
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Skipping compiled model.")
                self.compiled_model = None
        
        return self.compiled_model
    
    def export_to_onnx(self, force: bool = False) -> Path:
        """
        Export the model to ONNX format.
        
        Args:
            force: Force re-export even if ONNX model exists
            
        Returns:
            Path to exported ONNX model directory
        """
        if not self.config.export_onnx:
            logger.info("ONNX export disabled in configuration")
            return None
        
        if self.config.onnx_dir.exists() and not force:
            logger.info(f"ONNX model already exists at {self.config.onnx_dir}")
            return self.config.onnx_dir
        
        logger.info(f"Exporting model to ONNX format: {self.config.onnx_dir}")
        
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        
        try:
            self.onnx_model = ORTModelForSequenceClassification.from_pretrained(
                self.config.model_id,
                export=True,
                provider=provider,
                cache_dir=self.config.onnx_dir
            )
            logger.info(f"Model exported to ONNX successfully at {self.config.onnx_dir}")
            return self.config.onnx_dir
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise
    
    def load_onnx_model(self, model_dir: Optional[Path] = None) -> ORTModelForSequenceClassification:
        """
        Load ONNX model from directory.
        
        Args:
            model_dir: Directory containing ONNX model. If None, uses config.onnx_dir
            
        Returns:
            Loaded ONNX model
        """
        if model_dir is None:
            model_dir = self.config.onnx_dir
        
        if self.onnx_model is None:
            logger.info(f"Loading ONNX model from {model_dir}")
            provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
            
            try:
                self.onnx_model = ORTModelForSequenceClassification.from_pretrained(
                    model_dir,
                    provider=provider
                )
                logger.info("ONNX model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                raise
        
        return self.onnx_model
    
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        info = {
            "model_id": self.config.model_id,
            "device": self.device,
            "pytorch_loaded": self.pytorch_model is not None,
            "onnx_loaded": self.onnx_model is not None,
            "compiled_loaded": self.compiled_model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
        }
        
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        
        return info
