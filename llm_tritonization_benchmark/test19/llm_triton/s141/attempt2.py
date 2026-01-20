import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= n:
        return
        
    k = i * (i + 1) // 2
    
    for j in range(i, n):
        bb_offset = j * n + i
        bb_val = tl.load(bb_ptr + bb_offset)
        
        current_val = tl.load(flat_2d_array_ptr + k)
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        k += j + 1

def s141_triton(bb, flat_2d_array):
    n = bb.shape[0]
    
    grid = (n,)
    
    s141_kernel[grid](bb, flat_2d_array, n, BLOCK_SIZE=1)