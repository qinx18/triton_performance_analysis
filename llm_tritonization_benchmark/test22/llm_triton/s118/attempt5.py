import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is row-major, so bb[j][i] = bb[j * N + i_val]
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_valid_mask = mask & (a_read_offsets >= 0)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_valid_mask, other=0.0)
    
    # Compute products (mask bb_vals to match a_vals masking)
    bb_masked = tl.where(a_valid_mask, bb_vals, 0.0)
    products = bb_masked * a_vals
    result = tl.sum(products, axis=0)
    
    # Add to a[i]
    current_val = tl.load(a_ptr + i_val)
    tl.store(a_ptr + i_val, current_val + result)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    for i in range(1, N):
        grid = (1,)
        s118_kernel[grid](a, bb, i, N=N, BLOCK_SIZE=BLOCK_SIZE)