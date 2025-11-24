#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s343_baseline import s343_pytorch
from llm_triton.s343_triton_correct import s343_triton

# Small test case for debugging
N = 4
torch.manual_seed(42)

aa = torch.randn(N, N, device='cuda', dtype=torch.float32)
bb = torch.randn(N, N, device='cuda', dtype=torch.float32)
flat_2d_array = torch.zeros(N * N, device='cuda', dtype=torch.float32)

print("Input aa:")
print(aa)
print("\nInput bb:")
print(bb)
print("\nMask (bb > 0):")
print(bb > 0)
print()

# Run PyTorch baseline
flat_pytorch = flat_2d_array.clone()
pytorch_result = s343_pytorch(aa.clone(), bb.clone(), flat_pytorch)
print("PyTorch result:", pytorch_result[:10])
print()

# Try to run Triton
try:
    flat_triton = flat_2d_array.clone()
    triton_result = s343_triton(aa.clone(), bb.clone(), flat_triton)
    print("Triton result:", triton_result[:10])
    print()

    # Show differences
    print("Difference:", pytorch_result - triton_result)
    print("Max error:", torch.max(torch.abs(pytorch_result - triton_result)).item())
except Exception as e:
    print(f"Triton error: {e}")
    import traceback
    traceback.print_exc()
