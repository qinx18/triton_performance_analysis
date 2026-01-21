import triton
import triton.language as tl
import torch

@triton.jit
def s118_kernel(a_ptr, bb_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < i_val
    
    # Load bb[j][i] values - bb is stored row-major
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Load a[i-j-1] values
    a_read_offsets = i_val - j_offsets - 1
    a_read_mask = mask & (a_read_offsets >= 0) & (a_read_offsets < N)
    a_vals = tl.load(a_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Compute products
    products = tl.where(a_read_mask, bb_vals * a_vals, 0.0)
    
    # Sum reduction
    reduction = tl.sum(products, axis=0)
    
    # Add to a[i]
    a_i = tl.load(a_ptr + i_val)
    a_i = a_i + reduction
    tl.store(a_ptr + i_val, a_i)

def s118_triton(a, bb):
    N = bb.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, N):
        grid = (1,)
        s118_kernel[grid](
            a, bb, i, N, BLOCK_SIZE=BLOCK_SIZE
        )