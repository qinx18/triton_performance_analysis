#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s424_baseline_correct import s424_pytorch
from llm_triton.s424_triton_correct import s424_triton

# Small test case for debugging
N = 10
torch.manual_seed(42)

a = torch.randn(N, device='cuda', dtype=torch.float32)
flat_2d_array = torch.randn(200, device='cuda', dtype=torch.float32)  # Large enough

print("Understanding s424:")
print("="*60)
print("Original C code:")
print("  int vl = 63;")
print("  xx = flat_2d_array + vl;  // xx points to flat_2d_array[63]")
print("  for (int i = 0; i < LEN_1D - 1; i++) {")
print("      xx[i+1] = flat_2d_array[i] + a[i];")
print("  }")
print()
print("This means: flat_2d_array[63 + i + 1] = flat_2d_array[i] + a[i]")
print("Or: flat_2d_array[64 + i] = flat_2d_array[i] + a[i]")
print("="*60)
print()

print(f"Input a (size {N}):", a)
print(f"Input flat_2d_array[0:10]:", flat_2d_array[0:10])
print(f"Input flat_2d_array[64:74]:", flat_2d_array[64:74])
print()

# Run PyTorch baseline
flat_pytorch = flat_2d_array.clone()
pytorch_result = s424_pytorch(a, flat_pytorch)
print("After PyTorch:")
print(f"  flat_2d_array[0:10]:", pytorch_result[0:10])
print(f"  flat_2d_array[64:74]:", pytorch_result[64:74])
print()

# Run Triton
flat_triton = flat_2d_array.clone()
triton_result = s424_triton(a, flat_triton)
print("After Triton:")
print(f"  flat_2d_array[0:10]:", triton_result[0:10])
print(f"  flat_2d_array[64:74]:", triton_result[64:74])
print()

# Verify
print("Verification:")
print(f"  flat_2d_array[64] should equal flat_2d_array[0] + a[0]")
print(f"  Expected: {flat_2d_array[0].item():.4f} + {a[0].item():.4f} = {(flat_2d_array[0] + a[0]).item():.4f}")
print(f"  PyTorch:  {pytorch_result[64].item():.4f}")
print(f"  Triton:   {triton_result[64].item():.4f}")
print()

# Compare results
max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
print(f"Max error between PyTorch and Triton: {max_error:.2e}")

# Check where the differences are
diff = torch.abs(pytorch_result - triton_result)
if max_error > 1e-5:
    print("\nDifferences found at:")
    nonzero_diff = torch.nonzero(diff > 1e-5).squeeze()
    print(f"  Indices: {nonzero_diff[:20]}")  # Show first 20
