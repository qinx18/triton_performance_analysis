import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, num_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    j_mask = j_offsets < num_j
    
    bb_offsets = j_offsets * 256 + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    a_indices = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=j_mask, other=0.0)
    
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)
    
    if pid == 0:
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    for i in range(1, LEN_2D):
        num_j = i
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, i, num_j, BLOCK_SIZE=BLOCK_SIZE)
    
    return a