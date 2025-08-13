#!/usr/bin/env python3
import torch
import torch.nn as nn
import os
import json
import argparse
import time
import warnings
from model import CNN

def check_quantization_support():
    """Check if quantization is properly supported"""
    try:
        # try to create a simple quantized linear layer
        test_linear = nn.Linear(10, 5)
        from torch.quantization import quantize_dynamic
        quantize_dynamic(test_linear, {nn.Linear}, dtype=torch.qint8)
        return True, "dynamic"
    except Exception as e:
        print(f"Dynamic quantization not available: {e}")
        return False, None

def load_model(model_path, num_classes=2):
    """Load the trained model"""
    model = CNN(num_classes=num_classes)
    
    # handle different checkpoint formats
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def get_file_size_mb(filepath):
    """Get file size in MB"""
    size_bytes = os.path.getsize(filepath)
    return size_bytes / (1024 * 1024)

def benchmark_model(model, num_runs=100, input_size=(1, 3, 224, 224)):
    """Benchmark model inference time"""
    dummy_input = torch.randn(*input_size)
    
    # warmup
    with torch.no_grad():
        for _ in range(10):
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Warning: Model forward pass failed during warmup: {e}")
                return float('inf')
    
    # benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time_ms = (end_time - start_time) / num_runs * 1000
    return avg_time_ms

def create_pruned_model(model, pruning_ratio=0.3):
    """create pruned vers of  model as alternative to quantization"""
    import torch.nn.utils.prune as prune
    
    print(f"applying {pruning_ratio:.1%} unstructured pruning...")
    
    # createcopy of model
    pruned_model = type(model)(model.num_classes if hasattr(model, 'num_classes') else 2)
    pruned_model.load_state_dict(model.state_dict())
    
    # apply pruning to linear and conv layers
    modules_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            modules_to_prune.append((module, 'weight'))
    
    if modules_to_prune:
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # remove pruning masks to make permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
    
    return pruned_model

def half_precision_model(model):
    """convert model to half precision (FP16)"""
    print("Converting model to half precision (FP16)...")
    return model.half()

def quantize_model_advanced(model_path, output_path, num_classes=2):
    """advanced quantization with multiple fallback strategies"""
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, num_classes)
    
    # get original model info
    original_size = get_file_size_mb(model_path)
    original_time = benchmark_model(model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original inference time: {original_time:.2f} ms")
    
    # check quantization support
    quant_supported, quant_type = check_quantization_support()
    
    compressed_model = None
    compression_method = None
    
    if quant_supported:
        try:
            print("\nAttempting dynamic quantization...")
            from torch.quantization import quantize_dynamic
            
            # set quantization engine (try different engines)
            engines = ['fbgemm', 'qnnpack', 'onednn']
            quantization_success = False
            
            for engine in engines:
                try:
                    if hasattr(torch.backends, 'quantized') and hasattr(torch.backends.quantized, 'engine'):
                        torch.backends.quantized.engine = engine
                        print(f"Trying quantization engine: {engine}")
                    
                    compressed_model = quantize_dynamic(
                        model,
                        {nn.Linear},  # only quantize Linear layers
                        dtype=torch.qint8
                    )
                    
                    # test the quantized model
                    test_time = benchmark_model(compressed_model)
                    if test_time != float('inf'):
                        compression_method = f"Dynamic Quantization ({engine})"
                        quantization_success = True
                        break
                    
                except Exception as e:
                    print(f"Engine {engine} failed: {e}")
                    continue
            
            if not quantization_success:
                raise Exception("All quantization engines failed")
                
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
            compressed_model = None
    
    # Fallback 1: Try pruning
    if compressed_model is None:
        try:
            print("\nFalling back to model pruning...")
            compressed_model = create_pruned_model(model, pruning_ratio=0.3)
            compression_method = "Pruning (30%)"
        except Exception as e:
            print(f"Pruning failed: {e}")
            compressed_model = None
    
    # Fallback 2: Try half precision
    if compressed_model is None:
        try:
            print("\nFalling back to half precision...")
            compressed_model = half_precision_model(model.clone())
            compression_method = "Half Precision (FP16)"
        except Exception as e:
            print(f"Half precision failed: {e}")
            compressed_model = None
    
    # Fallback 3: Just optimize the model structure
    if compressed_model is None:
        print("\nUsing TorchScript optimization as final fallback...")
        try:
            compressed_model = torch.jit.script(model)
            compression_method = "TorchScript Optimization"
        except Exception as e:
            print(f"TorchScript failed: {e}")
            compressed_model = model  # Use original model
            compression_method = "No Compression (Original)"
    
    # Benchmark compressed model
    if compressed_model is not None:
        compressed_time = benchmark_model(compressed_model)
    else:
        compressed_model = model
        compressed_time = original_time
        compression_method = "No Compression (Original)"
    
    # Save the model
    success = False
    save_format = None
    
    # Try different save formats
    save_attempts = [
        ('.pt', lambda m, p: torch.jit.save(torch.jit.script(m), p)),
        ('.pth', lambda m, p: torch.save({
            'model_state_dict': m.state_dict() if hasattr(m, 'state_dict') else model.state_dict(),
            'model_class': 'CNN',
            'num_classes': num_classes,
            'compression_method': compression_method,
            'original_model_path': model_path
        }, p))
    ]
    
    for ext, save_func in save_attempts:
        try:
            current_output = output_path.replace('.pt', ext).replace('.pth', ext)
            save_func(compressed_model, current_output)
            output_path = current_output
            save_format = f"{'TorchScript' if ext == '.pt' else 'State dict'} ({ext})"
            success = True
            break
        except Exception as e:
            print(f"Failed to save as {ext}: {e}")
    
    if not success:
        raise Exception("Failed to save model in any format")
    
    compressed_size = get_file_size_mb(output_path)
    
    # Calculate improvements
    size_reduction = (original_size - compressed_size) / original_size if original_size > 0 else 0
    speed_improvement = original_time / compressed_time if compressed_time > 0 else 1.0
    
    # Print results
    print(f"\n{'='*60}")
    print("MODEL COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"Original:")
    print(f"  Size: {original_size:.2f} MB")
    print(f"  Inference time: {original_time:.2f} ms")
    print(f"\nCompressed:")
    print(f"  Size: {compressed_size:.2f} MB ({size_reduction:.1%} reduction)")
    print(f"  Inference time: {compressed_time:.2f} ms ({speed_improvement:.1f}x faster)")
    print(f"  Method: {compression_method}")
    print(f"  Format: {save_format}")
    print(f"  Saved to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.replace('.pt', '_info.json').replace('.pth', '_info.json')
    metadata = {
        'original_model_path': model_path,
        'compressed_model_path': output_path,
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'size_reduction_percent': size_reduction * 100,
        'original_inference_time_ms': original_time,
        'compressed_inference_time_ms': compressed_time,
        'speed_improvement': speed_improvement,
        'compression_method': compression_method,
        'save_format': save_format,
        'num_classes': num_classes,
        'quantization_supported': quant_supported
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    return compressed_model, output_path

def main():
    parser = argparse.ArgumentParser(description="Advanced model compression with fallbacks")
    parser.add_argument("--model", default="best.pth", 
                       help="Path to model weights")
    parser.add_argument("--output", default="model_compressed.pt", 
                       help="Output path for compressed model")
    parser.add_argument("--classes", type=int, default=2, 
                       help="Number of classes (default: 2)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        compressed_model, output_path = quantize_model_advanced(
            args.model, 
            args.output, 
            args.classes
        )
        print(f"\nModel compression completed successfully!")
        print(f"Use the compressed model: {output_path}")
        
        # Provide usage instructions
        print(f"\n{'='*60}")
        print("USAGE INSTRUCTIONS")
        print(f"{'='*60}")
        
        if output_path.endswith('.pt'):
            print("To load the TorchScript model:")
            print(f"model = torch.jit.load('{output_path}')")
            print("model.eval()")
        else:
            print("To load the compressed model:")
            print(f"from model import CNN")
            print(f"checkpoint = torch.load('{output_path}')")
            print(f"model = CNN(num_classes={args.classes})")
            print("model.load_state_dict(checkpoint['model_state_dict'])")
            print("model.eval()")
        
    except Exception as e:
        print(f"Error during compression: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()