#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s452_baseline import s452_pytorch
from llm_triton.s452_triton_llm import s452_triton

# Test with large array
N = 10000
torch.manual_seed(42)

a = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)
c = torch.randn(N, device='cuda', dtype=torch.float32)

# Run both
pytorch_result = s452_pytorch(a.clone(), b.clone(), c.clone())
triton_result = s452_triton(a.clone(), b.clone(), c.clone())

# Check errors
diff = torch.abs(pytorch_result - triton_result)
max_error = torch.max(diff).item()
mean_error = torch.mean(diff).item()

print(f"Max error: {max_error:.2e}")
print(f"Mean error: {mean_error:.2e}")
print()

# Find where largest errors occur
top_errors = torch.topk(diff, 10)
print("Top 10 error locations and values:")
for i, (err, idx) in enumerate(zip(top_errors.values, top_errors.indices)):
    idx_val = idx.item()
    print(f"  #{i+1}: index={idx_val}, error={err:.2e}")
    print(f"       PyTorch: {pytorch_result[idx_val]:.6f}, Triton: {triton_result[idx_val]:.6f}")
    print(f"       b[{idx_val}]={b[idx_val]:.6f}, c[{idx_val}]={c[idx_val]:.6f}, i+1={idx_val+1}")
    # Check the computation
    expected = b[idx_val] + c[idx_val] * (idx_val + 1)
    print(f"       Expected: {expected:.6f}")
    print()
