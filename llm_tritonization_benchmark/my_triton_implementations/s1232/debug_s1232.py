#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s1232_baseline import s1232_pytorch
from llm_triton.s1232_triton_correct import s1232_triton

# Small test
N = 5
aa = torch.ones(N, N, device='cuda', dtype=torch.float32)
bb = torch.arange(N*N, device='cuda', dtype=torch.float32).reshape(N, N)
cc = torch.zeros(N, N, device='cuda', dtype=torch.float32)

print("Input bb:")
print(bb)

# Run PyTorch baseline
pytorch_result = s1232_pytorch(aa.clone(), bb.clone(), cc.clone())
print("\nPyTorch result (should be bb for triangular i>=j):")
print(pytorch_result)

# Run Triton
triton_result = s1232_triton(aa.clone(), bb.clone(), cc.clone())
print("\nTriton result:")
print(triton_result)

print("\nDifference:")
print(pytorch_result - triton_result)

print("\nMax error:", torch.max(torch.abs(pytorch_result - triton_result)).item())
