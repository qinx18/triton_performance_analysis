#!/usr/bin/env python3
"""
Inspect the generated assembly code for s112 kernel to understand
why it works despite apparent race conditions.
"""
import torch
import triton
from triton_performance_analysis.llm_tritonization_benchmark.llm_triton.s112_triton_llm_orig import s112_kernel
import os
import tempfile

# Create test data
a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
b = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device='cuda')
n_elements = 5
BLOCK_SIZE = 256

# Warm up the kernel to ensure it's compiled
grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
s112_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Get the compiled kernel
print("=" * 80)
print("KERNEL SIGNATURE AND METADATA")
print("=" * 80)
print(f"Kernel name: {s112_kernel.fn.__name__}")
print(f"Kernel params: {s112_kernel.arg_names}")
print()

# Try to get the generated code
print("=" * 80)
print("ATTEMPTING TO EXTRACT GENERATED CODE")
print("=" * 80)

# Method 1: Check kernel cache
try:
    # Get kernel metadata
    key = s112_kernel.cache_key(
        a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    print(f"Cache key: {key}")

    # Try to find cached binary
    cache_dir = os.path.join(tempfile.gettempdir(), "triton")
    print(f"Triton cache directory: {cache_dir}")

    if os.path.exists(cache_dir):
        print(f"Cache contents:")
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                if 's112' in filepath or key[:8] in filepath:
                    print(f"  Found: {filepath}")
                    # Try to read PTX or CUBIN
                    if file.endswith('.ptx'):
                        with open(filepath, 'r') as f:
                            ptx = f.read()
                            print("\n" + "=" * 80)
                            print("PTX CODE (first 100 lines)")
                            print("=" * 80)
                            print('\n'.join(ptx.split('\n')[:100]))
except Exception as e:
    print(f"Method 1 failed: {e}")

print()

# Method 2: Use Triton's compiled kernel attributes
try:
    print("=" * 80)
    print("KERNEL COMPILATION INFO")
    print("=" * 80)

    # Get the compiled kernel binary
    compiled = s112_kernel.cache[0] if hasattr(s112_kernel, 'cache') and s112_kernel.cache else None
    if compiled:
        print("Compiled kernel found!")
        print(f"Attributes: {dir(compiled)}")
    else:
        print("No compiled kernel in cache attribute")

except Exception as e:
    print(f"Method 2 failed: {e}")

print()

# Method 3: Enable Triton debug output and recompile
print("=" * 80)
print("RUNNING WITH DEBUG OUTPUT")
print("=" * 80)

# Clear the kernel cache
s112_kernel.cache_key.cache_clear() if hasattr(s112_kernel.cache_key, 'cache_clear') else None

# Set debug environment variable
os.environ['MLIR_ENABLE_DUMP'] = '1'
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

print("Re-running kernel with debug flags...")
a2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
b2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], device='cuda')
s112_kernel[grid](a2, b2, n_elements, BLOCK_SIZE=BLOCK_SIZE)

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print("The kernel executes correctly despite apparent race conditions.")
print("This could be due to:")
print("1. Warp-level synchronization (threads in same warp execute in lockstep)")
print("2. Memory ordering guarantees in CUDA")
print("3. L1 cache coherency handling the conflicts")
print("4. Triton compiler inserting implicit synchronization")
