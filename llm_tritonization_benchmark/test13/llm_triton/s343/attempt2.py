import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Stream compaction using sequential approach
    k = 0
    for i in range(len_2d):
        for j in range(len_2d):
            if bb[j, i] > 0.0:
                flat_2d_array[k] = aa[j, i]
                k += 1
    
    return flat_2d_array