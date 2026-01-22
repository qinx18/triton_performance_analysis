import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute base j values (j starts from 0, increments by 1 for each i)
    base_j_vals = i_offsets
    
    # First store: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + base_j_vals, result1, mask=mask)
    
    # For conditional store: when c[i] > 0, j gets incremented and a[j] = c[i] + d[i] * e[i]
    # We need to calculate how many previous elements had c > 0 to get correct j offset
    c_positive = c_vals > 0.0
    
    # Calculate cumulative sum of positive c values up to current position
    cum_sum = tl.zeros_like(offsets, dtype=tl.int32)
    for shift in [1, 2, 4, 8, 16, 32, 64, 128]:
        if shift < BLOCK_SIZE:
            shifted_mask = c_positive
            shifted_vals = tl.where(shifted_mask, 1, 0)
            # Shift and add
            shifted_offsets = offsets - shift
            valid_shift = shifted_offsets >= 0
            shifted_cum = tl.where(valid_shift, cum_sum, 0)
            cum_sum = cum_sum + tl.where(valid_shift, shifted_vals, 0)
    
    # Add current element if c > 0
    cum_sum = cum_sum + tl.where(c_positive, 1, 0)
    
    # j offset for conditional store
    conditional_j_vals = base_j_vals + cum_sum
    
    # Store conditional result
    cond_mask = mask & c_positive
    result2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + conditional_j_vals, result2, mask=cond_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE
    )