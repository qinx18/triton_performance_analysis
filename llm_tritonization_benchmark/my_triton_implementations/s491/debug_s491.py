#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from baselines.s491_baseline_correct import s491_pytorch
from llm_triton.s491_triton_correct import s491_triton

# Small test case
N = 20
torch.manual_seed(42)

a = torch.zeros(N, device='cuda', dtype=torch.float32)
b = torch.arange(1, N+1, device='cuda', dtype=torch.float32)
c = torch.ones(N, device='cuda', dtype=torch.float32)
d = torch.ones(N, device='cuda', dtype=torch.float32)
ip = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 5, 5, 13, 14, 15, 16, 17, 18, 19], device='cuda', dtype=torch.int64)

print("Testing scatter with duplicate indices")
print(f"ip has duplicates at index 5: {ip}")
print(f"b values: {b}")
print(f"Result should be: a[ip[i]] = b[i] + c[i] * d[i] = b[i] + 1")
print(f"For duplicate index 5, last write should win (i=12 writes b[12]+1=13)")
print()

# Run PyTorch
a_pytorch = a.clone()
pytorch_result = s491_pytorch(a_pytorch, b, c, d, ip)
print(f"PyTorch result: {pytorch_result}")
print(f"PyTorch a[5] = {pytorch_result[5]:.1f} (should be 13 from last write)")
print()

# Run Triton
a_triton = a.clone()
triton_result = s491_triton(a_triton, b, c, d, ip)
print(f"Triton result:  {triton_result}")
print(f"Triton a[5] = {triton_result[5]:.1f}")
print()

# Compare
diff = torch.abs(pytorch_result - triton_result)
print(f"Max difference: {torch.max(diff).item():.2e}")
print(f"Differences at: {torch.nonzero(diff > 1e-5).squeeze()}")
