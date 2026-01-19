import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    
    j_mask = j_offsets < i_val
    
    bb_offsets = j_offsets * 256 + i_val
    a_read_offsets = i_val - j_offsets - 1
    
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=j_mask, other=0.0)
    
    products = bb_vals * a_vals
    reduction = tl.sum(products)
    
    if tl.program_id(0) == 0:
        current_a = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_a + reduction)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        if i > 0:
            grid = (triton.cdiv(i, BLOCK_SIZE),)
            s118_kernel[grid](a, bb, i, BLOCK_SIZE=BLOCK_SIZE)