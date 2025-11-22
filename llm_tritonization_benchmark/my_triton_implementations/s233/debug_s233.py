#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s233_baseline import s233_pytorch
from llm_triton.s233_triton_correct import s233_triton

# Small test
N = 5
aa = torch.ones(N, N, device='cuda', dtype=torch.float32)
bb = torch.ones(N, N, device='cuda', dtype=torch.float32) * 2
cc = torch.arange(N*N, device='cuda', dtype=torch.float32).reshape(N, N)

print("Input aa:")
print(aa)
print("\nInput bb:")
print(bb)
print("\nInput cc:")
print(cc)

# Run PyTorch baseline
aa_pytorch, bb_pytorch = s233_pytorch(aa.clone(), bb.clone(), cc.clone())
print("\nPyTorch aa result:")
print(aa_pytorch)
print("\nPyTorch bb result:")
print(bb_pytorch)

# Run Triton
aa_triton, bb_triton = s233_triton(aa.clone(), bb.clone(), cc.clone())
print("\nTriton aa result:")
print(aa_triton)
print("\nTriton bb result:")
print(bb_triton)

print("\naa Difference:")
print(aa_pytorch - aa_triton)
print("\nbb Difference:")
print(bb_pytorch - bb_triton)

print("\nMax aa error:", torch.max(torch.abs(aa_pytorch - aa_triton)).item())
print("Max bb error:", torch.max(torch.abs(bb_pytorch - bb_triton)).item())
