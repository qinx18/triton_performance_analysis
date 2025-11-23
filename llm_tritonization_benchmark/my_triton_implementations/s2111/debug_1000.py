#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s2111_baseline import s2111_pytorch
from llm_triton.s2111_triton_diagonal import s2111_triton

N = 1000
aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

print("Testing with N=1000...")
print(f"Input has NaN: {torch.isnan(aa).any()}")
print(f"Input has Inf: {torch.isinf(aa).any()}")

# Run PyTorch
pytorch_result = s2111_pytorch(aa.clone())
print(f"PyTorch result has NaN: {torch.isnan(pytorch_result).any()}")
print(f"PyTorch result has Inf: {torch.isinf(pytorch_result).any()}")

# Run Triton
triton_result = s2111_triton(aa.clone())
print(f"Triton result has NaN: {torch.isnan(triton_result).any()}")
print(f"Triton result has Inf: {torch.isinf(triton_result).any()}")

if torch.isnan(triton_result).any():
    nan_locations = torch.where(torch.isnan(triton_result))
    print(f"NaN found at {len(nan_locations[0])} locations")
    print(f"First few NaN locations: {list(zip(nan_locations[0][:10].tolist(), nan_locations[1][:10].tolist()))}")
