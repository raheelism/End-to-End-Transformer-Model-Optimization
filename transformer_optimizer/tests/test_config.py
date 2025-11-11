"""
Unit tests for configuration module.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import OptimizationConfig


def test_config_creation():
    """Test basic configuration creation."""
    config = OptimizationConfig()
    assert config.model_id == "distilbert-base-uncased-finetuned-sst-2-english"
    assert config.batch_size == 16
    assert config.device == "auto"
    print("✓ Config creation test passed")


def test_config_from_dict():
    """Test configuration from dictionary."""
    config_dict = {
        "model_id": "bert-base-uncased",
        "batch_size": 32,
        "device": "cpu"
    }
    config = OptimizationConfig.from_dict(config_dict)
    assert config.model_id == "bert-base-uncased"
    assert config.batch_size == 32
    assert config.device == "cpu"
    print("✓ Config from dict test passed")


def test_config_to_dict():
    """Test configuration to dictionary conversion."""
    config = OptimizationConfig(model_id="test-model", batch_size=8)
    config_dict = config.to_dict()
    assert config_dict["model_id"] == "test-model"
    assert config_dict["batch_size"] == 8
    assert isinstance(config_dict["output_dir"], str)
    print("✓ Config to dict test passed")


def test_config_save_load():
    """Test configuration save and load."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        # Create and save config
        config1 = OptimizationConfig(
            model_id="test-model",
            batch_size=32,
            device="cpu"
        )
        config1.save(temp_path)
        
        # Load config
        config2 = OptimizationConfig.from_json(temp_path)
        
        # Verify
        assert config2.model_id == "test-model"
        assert config2.batch_size == 32
        assert config2.device == "cpu"
        print("✓ Config save/load test passed")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_config_directory_creation():
    """Test output directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = OptimizationConfig(
            output_dir=Path(tmpdir) / "test_outputs"
        )
        config.create_output_dirs()
        
        assert config.output_dir.exists()
        assert config.onnx_dir.exists()
        assert config.quantized_dir.exists()
        print("✓ Config directory creation test passed")


if __name__ == "__main__":
    print("Running configuration tests...")
    print("=" * 50)
    
    test_config_creation()
    test_config_from_dict()
    test_config_to_dict()
    test_config_save_load()
    test_config_directory_creation()
    
    print("=" * 50)
    print("All tests passed! ✓")
