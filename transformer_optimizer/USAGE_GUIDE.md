# Transformer Optimizer - Usage Guide

This guide provides step-by-step instructions for using the Transformer Optimizer toolkit.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [CLI Usage](#cli-usage)
5. [Python API](#python-api)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for GPU acceleration

### Install Dependencies

```bash
cd transformer_optimizer
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import transformers; import optimum; print('Installation successful!')"
```

## Quick Start

### 1. Run with Default Settings

The simplest way to get started:

```bash
cd transformer_optimizer
python -m src.cli
```

This will:
- Use DistilBERT model (fine-tuned on SST-2)
- Export to ONNX format
- Create quantized version
- Run benchmarks and evaluation
- Save results to `outputs/` directory

### 2. View Results

Check the results:

```bash
ls outputs/
cat outputs/results/results.json
```

## Configuration

### Using Configuration Files

Create a custom configuration file (e.g., `my_config.json`):

```json
{
  "model_id": "bert-base-uncased",
  "device": "cuda",
  "batch_size": 32,
  "max_length": 256,
  "quantization_approach": "dynamic",
  "num_benchmark_iterations": 10
}
```

Use it:

```bash
python -m src.cli --config my_config.json
```

### Pre-configured Templates

We provide several configuration templates:

- `configs/default_config.json` - Balanced settings
- `configs/cpu_optimized_config.json` - Optimized for CPU inference

```bash
python -m src.cli --config configs/cpu_optimized_config.json
```

## CLI Usage

### Basic Commands

```bash
# Optimize default model
python -m src.cli

# Specify a different model
python -m src.cli --model-id bert-base-uncased

# Use GPU
python -m src.cli --device cuda

# Custom output directory
python -m src.cli --output-dir my_results
```

### Optimization Options

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

### Benchmarking Options

```bash
# Custom batch size
python -m src.cli --batch-size 32

# More benchmark iterations for stable results
python -m src.cli --num-iterations 20

# Skip benchmarking (faster, no performance metrics)
python -m src.cli --no-benchmark

# Skip evaluation (faster, no accuracy metrics)
python -m src.cli --no-evaluation
```

### Dataset Options

```bash
# Use a different dataset
python -m src.cli --dataset-name glue --dataset-config mrpc

# Use smaller dataset split
python -m src.cli --dataset-split "validation[:10%]"
```

### Complete Example

```bash
python -m src.cli \
  --model-id distilbert-base-uncased \
  --device cuda \
  --batch-size 32 \
  --num-iterations 10 \
  --quantization-approach dynamic \
  --output-dir outputs/distilbert_optimized \
  --log-level INFO
```

## Python API

### Basic Usage

```python
from pathlib import Path
from src.config import OptimizationConfig
from src.optimizer import TransformerOptimizer

# Create configuration
config = OptimizationConfig(
    model_id="distilbert-base-uncased-finetuned-sst-2-english",
    output_dir=Path("outputs"),
    batch_size=16,
    device="auto"
)

# Create optimizer
optimizer = TransformerOptimizer(config)

# Run optimization
summary = optimizer.optimize()

# Display results
optimizer.print_results_table()

# Get speedup analysis
speedups = optimizer.get_speedup_analysis()
for model_name, speedup in speedups.items():
    print(f"{model_name}: {speedup:.2f}x faster")
```

### Advanced Usage

```python
from pathlib import Path
from src.config import OptimizationConfig
from src.model_manager import ModelManager
from src.quantizer import ModelQuantizer
from src.benchmark import Benchmarker

# Configuration
config = OptimizationConfig(
    model_id="bert-base-uncased",
    output_dir=Path("outputs/bert"),
    device="cuda",
    batch_size=32
)

# Initialize components
model_manager = ModelManager(config)
quantizer = ModelQuantizer(config)

# Load model
tokenizer = model_manager.load_tokenizer()
pytorch_model = model_manager.load_pytorch_model()

# Export to ONNX
onnx_path = model_manager.export_to_onnx()
print(f"ONNX model saved to: {onnx_path}")

# Quantize
quantized_path = quantizer.quantize_model(onnx_path)
print(f"Quantized model saved to: {quantized_path}")

# Benchmark
benchmarker = Benchmarker(config, tokenizer)

# Load dataset
texts, labels = benchmarker.load_dataset()

# Benchmark PyTorch model
@torch.no_grad()
def pytorch_predict(tokens):
    tokens = {k: v.to("cuda") for k, v in tokens.items()}
    logits = pytorch_model(**tokens).logits
    return logits.argmax(-1).cpu().tolist()

pytorch_results = benchmarker.run_full_benchmark(
    pytorch_predict,
    "PyTorch Model"
)
print(pytorch_results)
```

## Examples

### 1. Basic Usage Example

```bash
cd examples
python basic_usage.py
```

This demonstrates:
- Simple optimization workflow
- Default configuration
- Results visualization

### 2. Custom Model Example

```bash
cd examples
python custom_model.py
```

This shows:
- Using a different model (BERT)
- Custom dataset configuration
- Advanced settings

### 3. Inference Demo

```bash
cd examples
python inference_demo.py
```

This demonstrates:
- Running inference with optimized models
- Comparing predictions across formats
- Using the pipeline API

## Common Workflows

### Workflow 1: Quick Model Optimization

For quickly optimizing a model:

```bash
# 1. Run optimization
python -m src.cli --model-id your-model-id

# 2. Check results
cat outputs/results/summary.json

# 3. Use the optimized model
# ONNX model: outputs/onnx/
# Quantized model: outputs/quantized/
```

### Workflow 2: Production Deployment

For production use:

```bash
# 1. Optimize with more iterations for stable benchmarks
python -m src.cli \
  --model-id your-model-id \
  --num-iterations 20 \
  --device cuda \
  --batch-size 32

# 2. Validate accuracy
# Check accuracy in outputs/results/results.json

# 3. Deploy the best model
# Use ONNX or quantized model based on results
```

### Workflow 3: Model Comparison

To compare different models:

```bash
# Optimize model 1
python -m src.cli --model-id model1 --output-dir outputs/model1

# Optimize model 2
python -m src.cli --model-id model2 --output-dir outputs/model2

# Compare results
diff outputs/model1/results/results.json outputs/model2/results/results.json
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: Can't find the src module

**Solution**:
```bash
# Make sure you're in the transformer_optimizer directory
cd transformer_optimizer

# Use the correct command
python -m src.cli
```

### Issue: ONNX Export Fails

**Problem**: Model cannot be exported to ONNX

**Solution**:
- Ensure model supports ONNX export
- Update transformers: `pip install --upgrade transformers optimum`
- Try disabling ONNX: `python -m src.cli --no-onnx`

### Issue: torch.compile Fails

**Problem**: torch.compile not available

**Solution**:
- Requires PyTorch 2.0+: `pip install --upgrade torch`
- Disable it: `python -m src.cli --no-torch-compile`

### Issue: CUDA Out of Memory

**Problem**: GPU runs out of memory

**Solution**:
```bash
# Reduce batch size
python -m src.cli --batch-size 8 --device cuda

# Or use CPU
python -m src.cli --device cpu
```

### Issue: Slow Performance

**Problem**: Optimization is taking too long

**Solutions**:
```bash
# 1. Use smaller dataset split
python -m src.cli --dataset-split "validation[:5%]"

# 2. Reduce benchmark iterations
python -m src.cli --num-iterations 3

# 3. Skip evaluation
python -m src.cli --no-evaluation

# 4. Skip benchmarking
python -m src.cli --no-benchmark
```

### Issue: Low Accuracy After Quantization

**Problem**: Quantized model has lower accuracy

**Solution**:
- This is expected; typically <1% accuracy drop
- Try per-channel quantization: Edit config with `"per_channel": true`
- Use static quantization with calibration data
- Consider using a different quantization approach

## Performance Tips

### For CPU

```bash
# Use CPU-optimized configuration
python -m src.cli --config configs/cpu_optimized_config.json

# Or set thread count manually
# Edit config: "omp_num_threads": "4", "mkl_num_threads": "4"
```

### For GPU

```bash
# Use larger batch size
python -m src.cli --device cuda --batch-size 32

# Enable torch.compile
python -m src.cli --device cuda --use-torch-compile
```

### For Production

```bash
# Run comprehensive benchmarks
python -m src.cli \
  --num-iterations 20 \
  --num-warmup-runs 5 \
  --save-config production_config.json
```

## Next Steps

1. **Explore Examples**: Run all examples in the `examples/` directory
2. **Read Documentation**: Check the main [README.md](README.md) for detailed information
3. **Customize**: Create your own configuration files for specific use cases
4. **Integrate**: Use the Python API to integrate into your applications
5. **Optimize**: Experiment with different settings to find the best configuration

## Support

For issues or questions:
1. Check this guide and the main README
2. Review the examples
3. Open an issue on GitHub with:
   - Your command or code
   - Error messages
   - System information (OS, Python version, GPU)

---

Happy optimizing! 🚀
