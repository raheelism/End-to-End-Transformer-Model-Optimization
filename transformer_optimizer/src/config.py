"""
Configuration management for transformer optimization.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for transformer model optimization."""
    
    # Model configuration
    model_id: str = "distilbert-base-uncased-finetuned-sst-2-english"
    task_type: str = "sequence-classification"
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    onnx_dir: Path = field(default_factory=lambda: Path("outputs/onnx"))
    quantized_dir: Path = field(default_factory=lambda: Path("outputs/quantized"))
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Quantization configuration
    quantization_approach: str = "dynamic"  # "dynamic" or "static"
    per_channel: bool = False
    reduce_range: bool = True
    
    # Benchmarking configuration
    benchmark_enabled: bool = True
    batch_size: int = 16
    max_length: int = 128
    num_warmup_runs: int = 3
    num_benchmark_iterations: int = 8
    
    # Evaluation configuration
    evaluation_enabled: bool = True
    dataset_name: str = "glue"
    dataset_config: str = "sst2"
    dataset_split: str = "validation[:20%]"
    metric_name: str = "accuracy"
    
    # Export options
    export_onnx: bool = True
    export_quantized: bool = True
    
    # Additional optimization options
    use_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_fullgraph: bool = False
    
    # Threading configuration
    omp_num_threads: str = "1"
    mkl_num_threads: str = "1"
    
    # Logging
    log_level: str = "INFO"
    save_results: bool = True
    results_format: str = "json"  # "json" or "csv"
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.onnx_dir, str):
            self.onnx_dir = Path(self.onnx_dir)
        if isinstance(self.quantized_dir, str):
            self.quantized_dir = Path(self.quantized_dir)
    
    @classmethod
    def from_json(cls, json_path: str) -> "OptimizationConfig":
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Loaded configuration from {json_path}")
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OptimizationConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Convert Path objects to strings for JSON serialization
        config_dict['output_dir'] = str(self.output_dir)
        config_dict['onnx_dir'] = str(self.onnx_dir)
        config_dict['quantized_dir'] = str(self.quantized_dir)
        return config_dict
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    def create_output_dirs(self):
        """Create all necessary output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        self.quantized_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directories: {self.output_dir}")
