# Quick Reference Guide

## Installation

```bash
cd transformer_optimizer
pip install -r requirements.txt
python validate_installation.py
```

## Basic Commands

### Run with defaults
```bash
python -m src.cli
```

### Common options
```bash
# Custom model
python -m src.cli --model-id bert-base-uncased

# Use GPU
python -m src.cli --device cuda

# Custom batch size
python -m src.cli --batch-size 32

# Use config file
python -m src.cli --config configs/default_config.json

# Custom output directory
python -m src.cli --output-dir my_results
```

## Optimization Control

```bash
# Skip ONNX export
python -m src.cli --no-onnx

# Skip quantization
python -m src.cli --no-quantization

# Disable torch.compile
python -m src.cli --no-torch-compile

# Static quantization
python -m src.cli --quantization-approach static
```

## Benchmarking

```bash
# Skip benchmarking (faster)
python -m src.cli --no-benchmark

# Skip evaluation
python -m src.cli --no-evaluation

# More iterations (more accurate)
python -m src.cli --num-iterations 20

# Smaller dataset (faster)
python -m src.cli --dataset-split "validation[:5%]"
```

## Python API

### Basic usage
```python
from pathlib import Path
from src.config import OptimizationConfig
from src.optimizer import TransformerOptimizer

config = OptimizationConfig(
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    output_dir=Path("outputs"),
)

optimizer = TransformerOptimizer(config)
summary = optimizer.optimize()
optimizer.print_results_table()
```

### Configuration options
```python
config = OptimizationConfig(
    model_id="bert-base-uncased",
    device="cuda",
    batch_size=32,
    max_length=256,
    quantization_approach="dynamic",
    export_onnx=True,
    export_quantized=True,
    use_torch_compile=True,
    num_benchmark_iterations=10,
)
```

## Examples

```bash
# Basic example
cd examples && python basic_usage.py

# Custom model
cd examples && python custom_model.py

# Inference demo
cd examples && python inference_demo.py
```

## Configuration Files

### Create from template
```bash
cp configs/default_config.json my_config.json
# Edit my_config.json
python -m src.cli --config my_config.json
```

### Save current config
```bash
python -m src.cli --save-config my_generated_config.json
```

## Output Structure

```
outputs/
├── onnx/                  # ONNX model
├── quantized/             # Quantized model
└── results/
    ├── results.json       # Benchmark results
    ├── results.csv        # CSV format
    └── summary.json       # Complete summary
```

## Common Workflows

### Quick test
```bash
python -m src.cli --dataset-split "validation[:5%]" --num-iterations 3
```

### Production optimization
```bash
python -m src.cli \
  --model-id your-model \
  --device cuda \
  --batch-size 32 \
  --num-iterations 20 \
  --output-dir production_outputs
```

### CPU-optimized
```bash
python -m src.cli --config configs/cpu_optimized_config.json
```

## Troubleshooting

### Import errors
```bash
# Make sure you're in the right directory
cd transformer_optimizer
python -m src.cli
```

### CUDA out of memory
```bash
python -m src.cli --batch-size 8 --device cuda
# or use CPU
python -m src.cli --device cpu
```

### Slow execution
```bash
# Use smaller dataset
python -m src.cli --dataset-split "validation[:10%]"

# Reduce iterations
python -m src.cli --num-iterations 3

# Skip some steps
python -m src.cli --no-evaluation --no-benchmark
```

## Help

```bash
# CLI help
python -m src.cli --help

# Run tests
python tests/test_config.py

# Validate installation
python validate_installation.py
```

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| model_id | distilbert-base-uncased-finetuned-sst-2-english | Model identifier |
| device | auto | Device (auto/cpu/cuda) |
| batch_size | 16 | Batch size |
| max_length | 128 | Max sequence length |
| quantization_approach | dynamic | Quantization type |
| num_benchmark_iterations | 8 | Benchmark runs |
| dataset_split | validation[:20%] | Dataset portion |
| export_onnx | true | Export to ONNX |
| export_quantized | true | Create quantized model |
| use_torch_compile | true | Use torch.compile |

## Performance Tips

- **CPU**: Increase `omp_num_threads` and `mkl_num_threads` to number of cores
- **GPU**: Use larger `batch_size` (32 or 64)
- **Accuracy**: Increase `num_benchmark_iterations` (20+)
- **Speed**: Reduce `dataset_split` to smaller percentage

## Links

- [Full README](README.md)
- [Usage Guide](USAGE_GUIDE.md)
- [Changelog](CHANGELOG.md)
