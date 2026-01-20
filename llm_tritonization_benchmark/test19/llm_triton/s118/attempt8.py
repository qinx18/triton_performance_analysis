import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is row-major, so bb[j][i] = bb[j*N + i]
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_indices = i_val - j_offsets - 1
    valid_mask = mask & (a_indices >= 0)
    a_vals = tl.load(a_ptr + a_indices, mask=valid_mask, other=0.0)
    
    # Apply valid_mask to bb_vals as well
    bb_vals = tl.where(valid_mask, bb_vals, 0.0)
    
    # Compute products and sum
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)
    
    # Add to a[i]
    old_val = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, old_val + result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        grid = (1,)
        s118_kernel[grid](a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)