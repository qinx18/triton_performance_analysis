#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s2111_baseline import s2111_pytorch
from llm_triton.s2111_triton_diagonal import s2111_triton

N = 1000
aa = torch.randn(N, N, device='cuda', dtype=torch.float32)

# Run both
pytorch_result = s2111_pytorch(aa.clone())
triton_result = s2111_triton(aa.clone())

# Check where they differ (excluding Inf/NaN)
finite_mask = torch.isfinite(pytorch_result) & torch.isfinite(triton_result)
diff = torch.abs(pytorch_result - triton_result)
finite_diff = diff[finite_mask]

print(f"Finite values: {finite_mask.sum()} / {N*N}")
print(f"Max error (finite only): {finite_diff.max().item():.6e}")
print(f"PyTorch Inf locations: {torch.isinf(pytorch_result).sum()}")
print(f"Triton Inf locations: {torch.isinf(triton_result).sum()}")
print(f"Inf locations match: {torch.equal(torch.isinf(pytorch_result), torch.isinf(triton_result))}")
