import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j, i] values
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_offsets >= 0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        current_val = tl.load(a_ptr + i_val)
        tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    for i in range(1, N):
        s118_kernel[(1,)](
            a, bb, i, N,
            BLOCK_SIZE=BLOCK_SIZE
        )