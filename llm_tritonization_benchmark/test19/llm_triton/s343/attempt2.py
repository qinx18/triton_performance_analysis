import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Use PyTorch for correct stream compaction
    k = 0
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                flat_2d_array[k] = aa[j, i]
                k += 1