import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_read_mask = mask & (a_read_offsets >= 0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Update a[i]
    if tl.program_id(0) == 0:
        old_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, old_val + result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        s118_kernel[(1,)](
            a, bb, i, 
            LEN_2D=LEN_2D, 
            BLOCK_SIZE=BLOCK_SIZE
        )