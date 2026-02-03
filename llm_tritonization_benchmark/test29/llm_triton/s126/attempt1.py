import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid
    
    if i >= len_2d:
        return
    
    k = i * len_2d + 1
    
    for j in range(1, len_2d):
        bb_prev = tl.load(bb_ptr + (j - 1) * len_2d + i)
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        cc_val = tl.load(cc_ptr + j * len_2d + i)
        
        result = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + j * len_2d + i, result)
        
        k += 1
    
def s126_triton(bb, cc, flat_2d_array, len_2d):
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s126_kernel[grid](
        bb,
        cc, 
        flat_2d_array,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )