#!/usr/bin/env python
"""
Validation script to check if all dependencies are installed correctly.
"""

import sys
from pathlib import Path

print("=" * 70)
print("Transformer Optimizer - Installation Validation")
print("=" * 70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✓ Python version OK")

# Check required packages
required_packages = {
    "torch": "2.0.0",
    "transformers": "4.49.0",
    "optimum": "1.20.0",
    "datasets": "2.20.0",
    "evaluate": "0.4.0",
    "accelerate": "0.20.0",
    "numpy": "1.24.0",
    "pandas": "2.0.0",
    "onnx": "1.14.0",
    "onnxruntime": "1.15.0",
}

print("\n2. Checking Required Packages:")
print("-" * 70)

all_ok = True
for package, min_version in required_packages.items():
    try:
        if package == "optimum":
            # Special handling for optimum
            import optimum
            version = optimum.__version__
        else:
            module = __import__(package)
            version = module.__version__
        
        print(f"   ✓ {package:20} {version:15} (min: {min_version})")
    except ImportError:
        print(f"   ❌ {package:20} NOT INSTALLED (min: {min_version})")
        all_ok = False
    except AttributeError:
        print(f"   ⚠ {package:20} INSTALLED (version unknown)")

if not all_ok:
    print("\n❌ Some packages are missing. Install with:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

# Check CUDA availability
print("\n3. CUDA Support:")
print("-" * 70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA is available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available (CPU-only mode)")
except Exception as e:
    print(f"   ⚠ Could not check CUDA: {e}")

# Check source files
print("\n4. Checking Source Files:")
print("-" * 70)

source_files = [
    "src/__init__.py",
    "src/config.py",
    "src/model_manager.py",
    "src/quantizer.py",
    "src/benchmark.py",
    "src/optimizer.py",
    "src/cli.py",
]

base_path = Path(__file__).parent
for file in source_files:
    file_path = base_path / file
    if file_path.exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ❌ {file} NOT FOUND")
        all_ok = False

# Test import
print("\n5. Testing Module Imports:")
print("-" * 70)

try:
    sys.path.insert(0, str(base_path / "src"))
    from config import OptimizationConfig
    from model_manager import ModelManager
    from quantizer import ModelQuantizer
    from benchmark import Benchmarker
    from optimizer import TransformerOptimizer
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    all_ok = False

# Test configuration
print("\n6. Testing Configuration:")
print("-" * 70)

try:
    config = OptimizationConfig()
    print(f"   ✓ Configuration created successfully")
    print(f"   Default model: {config.model_id}")
    print(f"   Default device: {config.device}")
except Exception as e:
    print(f"   ❌ Configuration test failed: {e}")
    all_ok = False

# Final result
print("\n" + "=" * 70)
if all_ok:
    print("✓ All checks passed! Installation is valid.")
    print("\nYou can now run:")
    print("  python -m src.cli --help")
    print("  python examples/basic_usage.py")
    print("=" * 70)
    sys.exit(0)
else:
    print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 70)
    sys.exit(1)
