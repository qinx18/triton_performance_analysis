import triton
import triton.language as tl
import torch

def s341_triton(a, b):
    # Stream compaction: pack positive values from b into beginning of a
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        a[:num_packed] = packed_values