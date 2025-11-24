#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s341_baseline import s341_pytorch
from llm_triton.s341_triton_llm import s341_triton

# Small test case for debugging
N = 10
torch.manual_seed(42)

a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)

print("Input b:", b)
print("Input a:", a)
print()

# Run PyTorch baseline
a_pytorch = a.clone()
pytorch_result = s341_pytorch(a_pytorch, b)
print("PyTorch result:", pytorch_result)
print()

# Run Triton
a_triton = a.clone()
triton_result = s341_triton(a_triton, b)
print("Triton result:", triton_result)
print()

# Show differences
print("Difference:", pytorch_result - triton_result)
print("Max error:", torch.max(torch.abs(pytorch_result - triton_result)).item())
