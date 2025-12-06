import torch
import triton
import triton.language as tl

def s341_triton(a, b):
    # Stream compaction using PyTorch's boolean indexing
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    a[:num_packed] = packed_values