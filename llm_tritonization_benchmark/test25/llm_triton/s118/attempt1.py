import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, len_2d, i, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i
    
    # Load bb[j][i] values for all valid j
    bb_offsets = j_offsets * len_2d + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values for all valid j
    a_indices = i - j_offsets - 1
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products)
    
    # Add to a[i]
    if tl.program_id(0) == 0:
        current_a = tl.load(a_ptr + i)
        tl.store(a_ptr + i, current_a + result)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(1, len_2d):
        if i > 0:  # Only launch kernel if there are j values to process
            s118_kernel[(1,)](
                a, bb, len_2d, i, BLOCK_SIZE=BLOCK_SIZE
            )