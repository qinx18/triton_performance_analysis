#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s351_baseline import s351_pytorch
from llm_triton.s351_triton_correct import s351_triton

# Test with size that's NOT a multiple of 5
N = 103  # 103 = 20*5 + 3, so last 3 elements should be untouched
torch.manual_seed(42)

a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)
alpha = 2.5

print(f"Testing with N={N} (not a multiple of 5)")
print(f"Last 3 elements should be untouched")
print()

# Save original values
a_orig = a.clone()

# Run PyTorch baseline
a_pytorch = a.clone()
pytorch_result = s351_pytorch(a_pytorch, b.clone(), alpha)

# Run Triton
a_triton = a.clone()
triton_result = s351_triton(a_triton, b.clone(), alpha)

print("Original a[-5:]:", a_orig[-5:])
print("PyTorch  a[-5:]:", pytorch_result[-5:])
print("Triton   a[-5:]:", triton_result[-5:])
print()

# Check last 3 elements
print("Last 3 elements comparison:")
print("PyTorch == Original?", torch.allclose(pytorch_result[-3:], a_orig[-3:]))
print("Triton  == Original?", torch.allclose(triton_result[-3:], a_orig[-3:]))
print()

# Check difference
max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
print(f"Max error: {max_error:.2e}")
print(f"Max error in last 3: {torch.max(torch.abs(pytorch_result[-3:] - triton_result[-3:])).item():.2e}")
