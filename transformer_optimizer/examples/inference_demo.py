"""
Example demonstrating inference with optimized models.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import pipeline
from config import OptimizationConfig
from model_manager import ModelManager
from quantizer import ModelQuantizer


def main():
    """Run inference demo with different model formats."""
    print("=" * 80)
    print("Inference Demo - Comparing Model Formats")
    print("=" * 80)
    
    # Configuration
    config = OptimizationConfig(
        model_id="distilbert-base-uncased-finetuned-sst-2-english",
        output_dir=Path("outputs/inference_demo"),
    )
    
    # Initialize managers
    model_manager = ModelManager(config)
    
    # Load tokenizer and models
    tokenizer = model_manager.load_tokenizer()
    pytorch_model = model_manager.load_pytorch_model()
    
    # Sample texts
    samples = [
        "What a fantastic movie—performed brilliantly!",
        "This was a complete waste of time.",
        "I'm not sure how I feel about this one.",
        "The acting was superb and the plot was engaging.",
        "Terrible experience, would not recommend.",
    ]
    
    print("\n" + "=" * 80)
    print("Sample Predictions")
    print("=" * 80)
    
    # Create PyTorch pipeline
    device = 0 if model_manager.device == "cuda" else -1
    pt_pipe = pipeline(
        "sentiment-analysis",
        model=pytorch_model,
        tokenizer=tokenizer,
        device=device
    )
    
    print("\nPyTorch Model Predictions:")
    print("-" * 80)
    for i, text in enumerate(samples, 1):
        result = pt_pipe(text)[0]
        print(f"{i}. '{text}'")
        print(f"   → {result['label']} (confidence: {result['score']:.4f})")
    
    # If ONNX model exists, compare
    if config.onnx_dir.exists():
        print("\n" + "=" * 80)
        print("ONNX Model Comparison")
        print("=" * 80)
        
        try:
            onnx_model = model_manager.load_onnx_model()
            ort_pipe = pipeline(
                "sentiment-analysis",
                model=onnx_model,
                tokenizer=tokenizer,
                device=-1  # ONNX Runtime handles device
            )
            
            print("\nONNX Model Predictions:")
            print("-" * 80)
            for i, text in enumerate(samples, 1):
                pt_result = pt_pipe(text)[0]
                ort_result = ort_pipe(text)[0]
                match = "✓" if pt_result['label'] == ort_result['label'] else "✗"
                print(f"{i}. '{text}'")
                print(f"   PyTorch: {pt_result['label']} ({pt_result['score']:.4f})")
                print(f"   ONNX:    {ort_result['label']} ({ort_result['score']:.4f}) {match}")
        except Exception as e:
            print(f"Could not load ONNX model: {e}")
            print("Run optimization first to generate ONNX model.")
    
    # If quantized model exists, compare
    if config.quantized_dir.exists():
        print("\n" + "=" * 80)
        print("Quantized Model Comparison")
        print("=" * 80)
        
        try:
            quantizer = ModelQuantizer(config)
            quant_model = quantizer.load_quantized_model()
            quant_pipe = pipeline(
                "sentiment-analysis",
                model=quant_model,
                tokenizer=tokenizer,
                device=-1
            )
            
            print("\nQuantized Model Predictions:")
            print("-" * 80)
            for i, text in enumerate(samples, 1):
                pt_result = pt_pipe(text)[0]
                quant_result = quant_pipe(text)[0]
                match = "✓" if pt_result['label'] == quant_result['label'] else "✗"
                print(f"{i}. '{text}'")
                print(f"   PyTorch:   {pt_result['label']} ({pt_result['score']:.4f})")
                print(f"   Quantized: {quant_result['label']} ({quant_result['score']:.4f}) {match}")
        except Exception as e:
            print(f"Could not load quantized model: {e}")
            print("Run optimization first to generate quantized model.")
    
    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
