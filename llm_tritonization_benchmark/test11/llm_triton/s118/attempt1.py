import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < i_val
    
    # Load bb[j][i] values
    bb_indices = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_indices = i_val - j_offsets - 1
    a_read_mask = j_mask & (a_read_indices >= 0)
    a_vals = tl.load(a_ptr + a_read_indices, mask=a_read_mask, other=0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    masked_products = tl.where(a_read_mask, products, 0.0)
    result = tl.sum(masked_products)
    
    # Load current a[i] and add result
    current_a = tl.load(a_ptr + i_val)
    new_a = current_a + result
    tl.store(a_ptr + i_val, new_a)

def s118_triton(a, bb):
    LEN_2D = bb.shape[0]
    
    for i in range(1, LEN_2D):
        BLOCK_SIZE = triton.next_power_of_2(i)
        s118_kernel[(1,)](a, bb, i, LEN_2D, BLOCK_SIZE)