#!/usr/bin/env python3
"""
Test s115 at N=50 to see the dramatic failure
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from baselines.s115_baseline import s115_pytorch
from llm_triton.s115_triton_llm import s115_triton

N = 50
print(f"Testing N={N}")

# Set seed
torch.manual_seed(42)

# Initialize arrays
a = torch.randn(N, device='cuda', dtype=torch.float32)
aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

print(f"\nInitial a[:10] = {a[:10]}")
print(f"Initial aa[0,:10] = {aa[0,:10]}")

# Run PyTorch
print("\nRunning PyTorch baseline...")
pytorch_result = s115_pytorch(a.clone(), aa.clone())
print(f"PyTorch result[:10] = {pytorch_result[:10]}")
print(f"PyTorch result[-10:] = {pytorch_result[-10:]}")

# Run Triton
print("\nRunning Triton...")
triton_result = s115_triton(a.clone(), aa.clone())
print(f"Triton result[:10] = {triton_result[:10]}")
print(f"Triton result[-10:] = {triton_result[-10:]}")

# Compare
diff = pytorch_result - triton_result
abs_diff = torch.abs(diff)
max_error = torch.max(abs_diff).item()
mean_error = torch.mean(abs_diff).item()

print(f"\nMax error: {max_error:.6e}")
print(f"Mean error: {mean_error:.6e}")

# Check for NaN/Inf
print(f"\nPyTorch has NaN: {torch.any(torch.isnan(pytorch_result)).item()}")
print(f"PyTorch has Inf: {torch.any(torch.isinf(pytorch_result)).item()}")
print(f"Triton has NaN: {torch.any(torch.isnan(triton_result)).item()}")
print(f"Triton has Inf: {torch.any(torch.isinf(triton_result)).item()}")

# Show where largest errors occur
max_idx = torch.argmax(abs_diff).item()
print(f"\nLargest error at index {max_idx}:")
print(f"  PyTorch: {pytorch_result[max_idx]:.10e}")
print(f"  Triton:  {triton_result[max_idx]:.10e}")
print(f"  Diff:    {diff[max_idx]:.10e}")

# Show top 10 errors
top_errors, top_indices = torch.topk(abs_diff, min(10, N))
print(f"\nTop 10 errors:")
for i, (err, idx) in enumerate(zip(top_errors, top_indices)):
    print(f"  {i+1}. idx={int(idx):4d}: error={err:.6e}, pytorch={pytorch_result[int(idx)]:.6e}, triton={triton_result[int(idx)]:.6e}")
