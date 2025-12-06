import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Use PyTorch boolean indexing for correct stream compaction
    mask = bb > 0.0
    packed_values = aa[mask]
    num_packed = packed_values.numel()
    
    if num_packed > 0:
        flat_2d_array[:num_packed] = packed_values