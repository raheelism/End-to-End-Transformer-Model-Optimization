"""
Model quantization using ONNX Runtime.
"""

import logging
from pathlib import Path
from typing import Optional

from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import QuantizationConfig

from .config import OptimizationConfig

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Handles model quantization operations."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize ModelQuantizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config
        self.quantized_model: Optional[ORTModelForSequenceClassification] = None
        
        logger.info("ModelQuantizer initialized")
    
    def quantize_model(
        self,
        onnx_model_path: Path,
        output_path: Optional[Path] = None,
        force: bool = False
    ) -> Path:
        """
        Quantize an ONNX model.
        
        Args:
            onnx_model_path: Path to the ONNX model directory
            output_path: Path to save quantized model. If None, uses config.quantized_dir
            force: Force re-quantization even if quantized model exists
            
        Returns:
            Path to quantized model directory
        """
        if not self.config.export_quantized:
            logger.info("Quantization disabled in configuration")
            return None
        
        if output_path is None:
            output_path = self.config.quantized_dir
        
        if output_path.exists() and not force:
            logger.info(f"Quantized model already exists at {output_path}")
            return output_path
        
        logger.info(f"Quantizing model from {onnx_model_path} to {output_path}")
        
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create quantizer
            quantizer = ORTQuantizer.from_pretrained(onnx_model_path)
            
            # Configure quantization
            qconfig = QuantizationConfig(
                approach=self.config.quantization_approach,
                per_channel=self.config.per_channel,
                reduce_range=self.config.reduce_range,
            )
            
            logger.info(f"Quantization config: approach={self.config.quantization_approach}, "
                       f"per_channel={self.config.per_channel}, "
                       f"reduce_range={self.config.reduce_range}")
            
            # Perform quantization
            quantizer.quantize(
                model_input=onnx_model_path,
                quantization_config=qconfig,
                save_dir=output_path
            )
            
            logger.info(f"Model quantized successfully and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
            raise
    
    def load_quantized_model(
        self,
        quantized_model_path: Optional[Path] = None,
        device: str = "cpu"
    ) -> ORTModelForSequenceClassification:
        """
        Load a quantized model.
        
        Args:
            quantized_model_path: Path to quantized model directory. If None, uses config.quantized_dir
            device: Device to use for inference ("cpu" or "cuda")
            
        Returns:
            Loaded quantized model
        """
        if quantized_model_path is None:
            quantized_model_path = self.config.quantized_dir
        
        if self.quantized_model is None:
            logger.info(f"Loading quantized model from {quantized_model_path}")
            
            provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            
            try:
                self.quantized_model = ORTModelForSequenceClassification.from_pretrained(
                    quantized_model_path,
                    provider=provider
                )
                logger.info("Quantized model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load quantized model: {e}")
                raise
        
        return self.quantized_model
    
    def get_model_size(self, model_path: Path) -> dict:
        """
        Get the size of model files.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary with size information
        """
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return {"exists": False}
        
        total_size = 0
        file_sizes = {}
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                file_sizes[file_path.name] = size
        
        return {
            "exists": True,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "file_sizes": file_sizes
        }
