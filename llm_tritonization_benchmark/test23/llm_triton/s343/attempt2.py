import torch
import triton
import triton.language as tl

def s343_triton(aa, bb, flat_2d_array):
    N = aa.shape[0]
    
    # Stream compaction: pack elements where bb[j][i] > 0
    # Process in column-major order (i outer, j inner) to match C code
    k = 0
    for i in range(N):
        for j in range(N):
            if bb[j, i] > 0.0:
                flat_2d_array[k] = aa[j, i]
                k += 1
    
    return k