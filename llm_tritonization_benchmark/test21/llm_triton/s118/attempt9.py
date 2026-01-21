import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values - correct 2D indexing
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    valid_indices = (a_indices >= 0) & mask
    a_vals = tl.load(a_ptr + a_indices, mask=valid_indices, other=0.0)
    
    # Compute products and sum
    products = tl.where(valid_indices, bb_vals * a_vals, 0.0)
    result = tl.sum(products)
    
    # Atomic add to a[i] to handle potential race conditions
    tl.atomic_add(a_ptr + i_val, result)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s118_kernel[grid](a, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)