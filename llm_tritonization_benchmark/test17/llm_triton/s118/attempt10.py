import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i, N):
    BLOCK_SIZE = 256
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i
    
    # Load bb[j][i] values
    bb_offsets = j_offsets * N + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i - j_offsets - 1
    a_indices_mask = mask & (a_indices >= 0)
    a_vals = tl.load(a_ptr + a_indices, mask=a_indices_mask, other=0.0)
    
    # Compute products
    products = bb_vals * a_vals
    
    # Sum the products (with proper masking)
    masked_products = tl.where(a_indices_mask, products, 0.0)
    result = tl.sum(masked_products)
    
    # Load current a[i] and add the result
    current_a = tl.load(a_ptr + i)
    new_a = current_a + result
    tl.store(a_ptr + i, new_a)

def s118_triton(a, bb):
    N = bb.shape[0]
    
    for i in range(1, N):
        s118_kernel[(1,)](a, bb, tl.constexpr(i), tl.constexpr(N))