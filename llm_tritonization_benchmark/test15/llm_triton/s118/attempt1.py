import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, BLOCK_SIZE: tl.constexpr, LEN_2D: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    a_read_offsets = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_offsets >= 0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    if tl.program_id(0) == 0:
        current_a = tl.load(a_ptr + i_val)
        new_a = current_a + result
        tl.store(a_ptr + i_val, new_a)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        s118_kernel[(1,)](
            a, bb,
            i,
            BLOCK_SIZE=BLOCK_SIZE,
            LEN_2D=LEN_2D
        )