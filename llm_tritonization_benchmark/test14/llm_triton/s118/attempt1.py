import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * len_2d + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i - j_offsets - 1
    a_read_mask = j_mask & (a_read_offsets >= 0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    reduction_mask = j_mask
    sum_val = tl.sum(tl.where(reduction_mask, products, 0.0))
    
    # Load current a[i] and add the sum
    current_a = tl.load(a_ptr + i)
    new_val = current_a + sum_val
    tl.store(a_ptr + i, new_val)

def s118_triton(a, bb):
    len_2d = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(1, len_2d):
        if i > 0:  # Only launch kernel if there are j values to process
            s118_kernel[(1,)](a, bb, i, len_2d, BLOCK_SIZE)