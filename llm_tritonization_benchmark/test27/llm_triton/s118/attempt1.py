import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get block of j indices
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j, i] values
    bb_offsets = j_offsets * len_2d + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(tl.where(a_read_mask, products, 0.0))
    
    # Load current a[i] and add result
    current_a = tl.load(a_ptr + i_val)
    new_a = current_a + result
    tl.store(a_ptr + i_val, new_a)

def s118_triton(a, bb, len_2d):
    BLOCK_SIZE = 256
    
    # Sequential loop over i, parallel reduction over j
    for i in range(1, len_2d):
        if i > 0:  # Only launch kernel if there are j values to process
            s118_kernel[(1,)](
                a, bb, i, len_2d, BLOCK_SIZE=BLOCK_SIZE
            )