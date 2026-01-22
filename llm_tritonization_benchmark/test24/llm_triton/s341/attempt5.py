import triton
import triton.language as tl
import torch

def s341_triton(a, b):
    # Use PyTorch's boolean indexing for correct stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    a[:num_packed] = packed_values