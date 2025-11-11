# End-to-End Transformer Model Optimization

A comprehensive repository for production-grade transformer model optimization using Hugging Face Optimum, ONNX Runtime, and Quantization techniques.

## 🚀 Overview

This repository provides a complete solution for optimizing transformer models with:
- **ONNX Runtime** export and optimization
- **Dynamic and Static Quantization** for model compression
- **torch.compile** integration for PyTorch 2.0+
- **Comprehensive Benchmarking** and evaluation
- **Production-ready** code architecture

## 📁 Repository Structure

```
End-to-End-Transformer-Model-Optimization/
└── transformer_optimizer/       # Main optimization toolkit
    ├── src/                    # Source code
    ├── configs/                # Configuration templates
    ├── examples/               # Usage examples
    ├── tests/                  # Unit tests
    ├── requirements.txt        # Dependencies
    ├── setup.py               # Package setup
    └── README.md              # Detailed documentation
```

## 🎯 Quick Start

### Installation

```bash
cd transformer_optimizer
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run optimization with default settings
python -m src.cli

# Optimize a custom model
python -m src.cli --model-id bert-base-uncased --device cuda

# Use a configuration file
python -m src.cli --config configs/default_config.json
```

### Python API

```python
from pathlib import Path
from transformer_optimizer.src.config import OptimizationConfig
from transformer_optimizer.src.optimizer import TransformerOptimizer

# Create configuration
config = OptimizationConfig(
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    output_dir=Path("outputs"),
    batch_size=16,
)

# Run optimization
optimizer = TransformerOptimizer(config)
summary = optimizer.optimize()
optimizer.print_results_table()
```

## ✨ Features

- **Multiple Optimization Techniques**: ONNX, Quantization, torch.compile
- **Automatic Benchmarking**: Speed and accuracy evaluation
- **Flexible Configuration**: JSON configs and CLI options
- **Production Ready**: Modular design, logging, error handling
- **Easy to Use**: Simple API and comprehensive examples

## 📖 Documentation

For detailed documentation, configuration options, and advanced usage, see the [transformer_optimizer README](transformer_optimizer/README.md).

## 🎓 Examples

Check out the `transformer_optimizer/examples/` directory for:
- **basic_usage.py** - Simple optimization example
- **custom_model.py** - Custom model and dataset configuration
- **inference_demo.py** - Inference comparison across formats

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers >= 4.49.0
- optimum[onnxruntime] >= 1.20.0
- datasets >= 2.20.0
- evaluate >= 0.4.0

See `transformer_optimizer/requirements.txt` for complete dependencies.

## 📊 Expected Results

Typical optimization results for DistilBERT on sentiment analysis:

| Model Format | Inference Time | Speedup | Accuracy |
|--------------|---------------|---------|----------|
| PyTorch Eager | Baseline | 1.0x | 0.9080 |
| torch.compile | -20-30% | 1.3x | 0.9080 |
| ONNX Runtime | -30-50% | 1.5-2x | 0.9080 |
| ONNX Quantized | -50-70% | 2-3x | 0.9050+ |

*Results vary based on hardware and model architecture*

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization techniques
- Support for more model architectures
- Extended benchmarking capabilities
- Performance improvements

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built with:
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face Optimum](https://huggingface.co/docs/optimum/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch](https://pytorch.org/)

---

For detailed documentation and advanced usage, see the [transformer_optimizer README](transformer_optimizer/README.md).
