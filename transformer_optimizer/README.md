# End-to-End Transformer Model Optimization

A production-grade toolkit for optimizing Transformer models using Hugging Face Optimum, ONNX Runtime, and Quantization techniques. This project provides a comprehensive solution for model optimization with benchmarking, evaluation, and easy-to-use APIs.

## 🚀 Features

- **Multiple Optimization Techniques**
  - ONNX Runtime export and optimization
  - Dynamic and static quantization
  - PyTorch torch.compile support
  - Automatic device detection (CPU/CUDA)

- **Comprehensive Benchmarking**
  - Inference speed measurement
  - Accuracy evaluation
  - Speedup analysis
  - Memory usage tracking

- **Production-Ready**
  - Modular architecture
  - Configuration management
  - Extensive logging
  - CLI and Python API
  - Error handling

- **Easy to Use**
  - Simple CLI interface
  - Example scripts
  - Configuration templates
  - Detailed documentation

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🔧 Installation

### Install from source

```bash
cd transformer_optimizer
pip install -r requirements.txt
```

### Install as package

```bash
cd transformer_optimizer
pip install -e .
```

## 🎯 Quick Start

### Using the CLI

Optimize with default settings:

```bash
python -m src.cli
```

Optimize a custom model:

```bash
python -m src.cli --model-id bert-base-uncased --device cuda
```

Use a configuration file:

```bash
python -m src.cli --config configs/default_config.json
```

### Using Python API

```python
from pathlib import Path
from src.config import OptimizationConfig
from src.optimizer import TransformerOptimizer

# Create configuration
config = OptimizationConfig(
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    output_dir=Path("outputs"),
    batch_size=16,
)

# Create optimizer and run
optimizer = TransformerOptimizer(config)
summary = optimizer.optimize()

# Display results
optimizer.print_results_table()
```

## 📖 Usage Examples

### Basic Usage

```bash
cd examples
python basic_usage.py
```

This example demonstrates:
- Loading a pre-trained model
- Running optimization pipeline
- Benchmarking different formats
- Displaying results

### Custom Model Optimization

```bash
cd examples
python custom_model.py
```

This example shows:
- Using a different model
- Custom dataset configuration
- Advanced optimization settings

### Inference Demo

```bash
cd examples
python inference_demo.py
```

This example demonstrates:
- Running inference with optimized models
- Comparing predictions across formats
- Using pipeline API

## ⚙️ Configuration

### Configuration File

Create a JSON configuration file:

```json
{
  "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
  "device": "auto",
  "output_dir": "outputs",
  "batch_size": 16,
  "max_length": 128,
  "quantization_approach": "dynamic",
  "export_onnx": true,
  "export_quantized": true,
  "use_torch_compile": true,
  "benchmark_enabled": true,
  "evaluation_enabled": true
}
```

### Configuration Options

#### Model Configuration
- `model_id`: Hugging Face model identifier
- `task_type`: Type of task (sequence-classification, token-classification, etc.)
- `device`: Device to use ("auto", "cpu", "cuda")

#### Optimization Options
- `export_onnx`: Export model to ONNX format
- `export_quantized`: Create quantized version
- `quantization_approach`: "dynamic" or "static"
- `per_channel`: Enable per-channel quantization
- `reduce_range`: Reduce quantization range
- `use_torch_compile`: Enable torch.compile optimization

#### Benchmarking Options
- `benchmark_enabled`: Enable performance benchmarking
- `batch_size`: Batch size for inference
- `max_length`: Maximum sequence length
- `num_warmup_runs`: Number of warmup iterations
- `num_benchmark_iterations`: Number of benchmark iterations

#### Evaluation Options
- `evaluation_enabled`: Enable accuracy evaluation
- `dataset_name`: Dataset name (e.g., "glue")
- `dataset_config`: Dataset configuration (e.g., "sst2")
- `dataset_split`: Dataset split to use
- `metric_name`: Metric to compute (e.g., "accuracy")

#### Output Options
- `output_dir`: Base output directory
- `save_results`: Save results to files
- `results_format`: Format for results ("json", "csv")
- `log_level`: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")

## 📊 Output

### Results Structure

```
outputs/
├── onnx/                    # ONNX model files
│   ├── model.onnx
│   └── config.json
├── quantized/               # Quantized model files
│   ├── model_quantized.onnx
│   └── config.json
└── results/                 # Benchmark results
    ├── results.json         # Detailed results
    ├── results.csv          # CSV format
    └── summary.json         # Complete summary
```

### Results Format

```json
{
  "model_name": "PyTorch Eager",
  "mean_ms": 2000.5,
  "std_ms": 50.2,
  "min_ms": 1950.0,
  "max_ms": 2100.0,
  "accuracy": 0.9080
}
```

## 🏗️ Architecture

### Project Structure

```
transformer_optimizer/
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── model_manager.py    # Model loading/conversion
│   ├── quantizer.py        # Quantization logic
│   ├── benchmark.py        # Benchmarking utilities
│   ├── optimizer.py        # Main orchestration
│   └── cli.py              # Command-line interface
├── configs/                 # Configuration templates
│   ├── default_config.json
│   └── cpu_optimized_config.json
├── examples/                # Example scripts
│   ├── basic_usage.py
│   ├── custom_model.py
│   └── inference_demo.py
├── tests/                   # Unit tests
├── outputs/                 # Output directory (generated)
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
└── README.md               # This file
```

### Core Components

#### OptimizationConfig
Manages all configuration settings with validation and serialization.

#### ModelManager
Handles model loading, device management, and format conversion.

#### ModelQuantizer
Performs quantization using ONNX Runtime with configurable options.

#### Benchmarker
Measures inference performance and computes accuracy metrics.

#### TransformerOptimizer
Orchestrates the complete optimization pipeline.

## 🔬 Benchmarking Details

### Metrics Collected

1. **Performance Metrics**
   - Mean inference time (milliseconds)
   - Standard deviation
   - Min/max times
   - Throughput (samples/second)

2. **Accuracy Metrics**
   - Task-specific metrics (accuracy, F1, etc.)
   - Prediction consistency across formats

3. **Model Size**
   - Original model size
   - ONNX model size
   - Quantized model size
   - Compression ratio

### Optimization Techniques

1. **ONNX Runtime**
   - Graph optimizations
   - Operator fusion
   - Memory layout optimization
   - Hardware-specific acceleration

2. **Quantization**
   - Dynamic quantization (run-time)
   - Static quantization (calibration-based)
   - Per-channel vs per-tensor
   - INT8 quantization

3. **torch.compile**
   - Graph compilation
   - Operator fusion
   - Memory optimization
   - CUDA kernel optimization

## 🎛️ CLI Reference

### Basic Commands

```bash
# Optimize with defaults
python -m src.cli

# Specify model
python -m src.cli --model-id bert-base-uncased

# Use GPU
python -m src.cli --device cuda

# Custom output directory
python -m src.cli --output-dir my_outputs
```

### Optimization Control

```bash
# Skip ONNX export
python -m src.cli --no-onnx

# Skip quantization
python -m src.cli --no-quantization

# Use static quantization
python -m src.cli --quantization-approach static

# Disable torch.compile
python -m src.cli --no-torch-compile
```

### Benchmarking Control

```bash
# Custom batch size
python -m src.cli --batch-size 32

# More iterations
python -m src.cli --num-iterations 20

# Skip benchmarking
python -m src.cli --no-benchmark

# Skip evaluation
python -m src.cli --no-evaluation
```

### Configuration Management

```bash
# Load from config file
python -m src.cli --config my_config.json

# Save configuration
python -m src.cli --save-config generated_config.json
```

## 🔍 Advanced Topics

### Custom Models

To optimize custom models:

1. Ensure model is compatible with Hugging Face transformers
2. Model must support ONNX export
3. Provide appropriate task type in configuration

### Custom Datasets

To use custom datasets:

1. Dataset must be compatible with Hugging Face datasets
2. Specify correct dataset name and configuration
3. Ensure proper metric is selected

### Performance Tuning

For CPU:
```bash
# Increase thread count
python -m src.cli --device cpu
# Set in config: "omp_num_threads": "4", "mkl_num_threads": "4"
```

For GPU:
```bash
python -m src.cli --device cuda --batch-size 32
```

### Static Quantization

For static quantization with calibration:

```python
config = OptimizationConfig(
    quantization_approach="static",
    # Additional calibration settings...
)
```

## 📝 Best Practices

1. **Start with defaults**: Use default configuration first
2. **Benchmark systematically**: Run multiple iterations for reliable results
3. **Validate accuracy**: Always check accuracy after optimization
4. **Monitor memory**: Watch GPU/CPU memory usage
5. **Version control**: Save configurations for reproducibility
6. **Test inference**: Validate predictions before deployment

## 🐛 Troubleshooting

### Common Issues

**Issue**: ONNX export fails
- **Solution**: Ensure model supports ONNX export; try updating transformers/optimum

**Issue**: torch.compile fails
- **Solution**: Requires PyTorch 2.0+; disable with `--no-torch-compile`

**Issue**: Low speedup on CPU
- **Solution**: Try increasing thread count in configuration

**Issue**: Memory errors
- **Solution**: Reduce batch size or sequence length

## 📈 Performance Tips

1. **Use appropriate batch size**: Larger batches improve throughput but use more memory
2. **Enable torch.compile**: Can provide 2-3x speedup on compatible hardware
3. **Use quantization**: Typically 2-4x speedup with minimal accuracy loss
4. **Tune threads**: For CPU, set threads to number of physical cores
5. **Use GPU**: 10-100x speedup compared to CPU for larger models

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model architectures
- More optimization techniques
- Extended benchmarking metrics
- Better error handling
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformers and optimum libraries
- ONNX Runtime team for optimization tools
- PyTorch team for the deep learning framework

## 📚 References

- [Hugging Face Optimum](https://huggingface.co/docs/optimum/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This is a production-grade implementation based on best practices for transformer model optimization. Adjust configurations based on your specific use case and hardware.
