import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, num_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_start = pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    j_mask = j_offsets < num_j
    
    bb_row_indices = j_offsets
    bb_col_index = i_val
    bb_indices = bb_row_indices * 256 + bb_col_index
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    a_indices = i_val - j_offsets - 1
    a_mask = j_mask & (a_indices >= 0) & (a_indices < 32000)
    a_vals = tl.load(a_ptr + a_indices, mask=a_mask, other=0.0)
    
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)
    
    if result != 0.0:
        tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    for i in range(1, LEN_2D):
        num_j = i
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, i, num_j, BLOCK_SIZE=BLOCK_SIZE)
    
    return a