#!/usr/bin/env python3
"""
Extract PTX assembly from s112 kernel
"""
import torch
import triton
from triton_performance_analysis.llm_tritonization_benchmark.llm_triton.s112_triton_llm import s112_kernel

# Create test inputs
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')
n_elements = 1024
BLOCK_SIZE = 256

# Run kernel once to compile it
grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
s112_kernel[grid](a, b, n_elements, BLOCK_SIZE=BLOCK_SIZE)

# Access the compiled kernel
print("=" * 80)
print("EXTRACTING GENERATED CODE")
print("=" * 80)

# The kernel object has a `run` method that gets compiled
# We need to access the backend to get assembly
try:
    # Get the compiled binary
    from triton.runtime import driver

    # Kernel has been cached after first run
    # Try to access metadata
    if hasattr(s112_kernel, 'cache'):
        print(f"Cache has {len(s112_kernel.cache)} entries")

        for i, (key, compiled_kernel) in enumerate(s112_kernel.cache.items()):
            print(f"\n--- Kernel variant {i} ---")
            print(f"Key: {key}")

            # Try to get assembly from compiled kernel
            if hasattr(compiled_kernel, 'asm'):
                print("\nFound ASM:")
                print(compiled_kernel.asm)
            elif hasattr(compiled_kernel, 'module'):
                print("\nFound module")
                if hasattr(compiled_kernel.module, 'get_asm'):
                    print(compiled_kernel.module.get_asm())
            elif hasattr(compiled_kernel, '__dict__'):
                print(f"\nCompiled kernel attributes: {compiled_kernel.__dict__.keys()}")
                for attr_name in ['asm', 'ptx', 'sass', 'cubin']:
                    if hasattr(compiled_kernel, attr_name):
                        attr_val = getattr(compiled_kernel, attr_name)
                        print(f"\n{attr_name.upper()}:")
                        print(str(attr_val)[:2000])  # First 2000 chars
    else:
        print("No cache found")

except Exception as e:
    print(f"Error accessing compiled kernel: {e}")
    import traceback
    traceback.print_exc()

# Alternative: use Triton's debug environment variable
print("\n" + "=" * 80)
print("TRYING ALTERNATIVE METHOD")
print("=" * 80)

import os
import tempfile

# Set environment to dump generated code
cache_dir = os.path.join(tempfile.gettempdir(), "triton", "user")
os.makedirs(cache_dir, exist_ok=True)

print(f"Triton cache directory: {cache_dir}")

# List all files
if os.path.exists(cache_dir):
    all_files = []
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    print(f"\nFound {len(all_files)} cached files")

    # Look for PTX files
    ptx_files = [f for f in all_files if f.endswith('.ptx')]
    if ptx_files:
        print(f"\nFound {len(ptx_files)} PTX files")
        # Read the most recent one
        latest_ptx = max(ptx_files, key=os.path.getmtime)
        print(f"\nReading: {latest_ptx}")
        with open(latest_ptx, 'r') as f:
            ptx_code = f.read()
            print("\n" + "=" * 80)
            print("PTX CODE")
            print("=" * 80)
            print(ptx_code[:5000])  # First 5000 characters
    else:
        print("No PTX files found in cache")
        print(f"Available file types: {set(f.split('.')[-1] for f in all_files if '.' in f)}")
