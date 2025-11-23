#!/usr/bin/env python3
"""
Debug s2111 - check what's wrong with the implementation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s2111_baseline import s2111_pytorch
from llm_triton.s2111_triton_diagonal import s2111_triton

# Test with a small matrix
N = 5
aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
print("Input matrix:")
print(aa)
print()

# Run PyTorch baseline
pytorch_result = s2111_pytorch(aa.clone())
print("PyTorch result:")
print(pytorch_result)
print()

# Run Triton
triton_result = s2111_triton(aa.clone())
print("Triton result:")
print(triton_result)
print()

# Compare
diff = torch.abs(pytorch_result - triton_result)
print("Absolute difference:")
print(diff)
print()

max_error = torch.max(diff).item()
print(f"Max error: {max_error:.6f}")
