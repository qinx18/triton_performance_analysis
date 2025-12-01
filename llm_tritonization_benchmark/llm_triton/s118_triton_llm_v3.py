import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Vectorized j indices for this block
    j_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = (j_offsets < i_val) & (j_offsets >= 0)
    
    # Load bb[j][i] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0) & (a_read_indices < LEN_2D)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    masked_products = tl.where(j_mask & a_read_mask, products, 0.0)
    partial_sum = tl.sum(masked_products)
    
    # Atomic add to a[i]
    tl.atomic_add(a_ptr + i_val, partial_sum)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        num_j = i
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s118_kernel[grid](
                a, bb,
                i, LEN_2D, BLOCK_SIZE
            )