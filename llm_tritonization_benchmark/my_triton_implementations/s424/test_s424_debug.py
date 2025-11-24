#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from baselines.s424_baseline_correct import s424_pytorch
from llm_triton.s424_triton_correct import s424_triton

N = 100
torch.manual_seed(42)

# Initialize arrays
a = torch.randn(N + 10, device='cuda', dtype=torch.float32)
flat_2d_array = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)

# Save original
orig = flat_2d_array.clone()

# Run PyTorch baseline
pytorch_result = s424_pytorch(a.clone(), flat_2d_array.clone())

# Run Triton corrected
triton_result = s424_triton(a.clone(), flat_2d_array.clone())

print(f"PyTorch result type: {type(pytorch_result)}, shape: {pytorch_result.shape}")
print(f"Triton result type: {type(triton_result)}, shape: {triton_result.shape}")
print(f"Original flat_2d_array shape: {flat_2d_array.shape}")

# Compare results
max_error = torch.max(torch.abs(pytorch_result - triton_result)).item()
print(f"Max error: {max_error:.2e}")

# Check specific locations
print(f"\nChecking flat_2d_array[64:74]:")
print(f"  PyTorch: {pytorch_result[64:74]}")
print(f"  Triton:  {triton_result[64:74]}")
print(f"  Diff:    {pytorch_result[64:74] - triton_result[64:74]}")

# Find where differences occur
diff = torch.abs(pytorch_result - triton_result)
large_diff_indices = torch.nonzero(diff > 1e-3).squeeze()
print(f"\nIndices with large differences (> 1e-3): {large_diff_indices[:20]}")
