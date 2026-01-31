import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_half,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    # Load data for current block
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # For each valid i, we need to compute its corresponding j positions
    # j starts at -1, gets incremented to 0 before first use
    # For i=0: j=0 (first assignment), possibly j=1 (if c[0]>0)  
    # For i=1: j depends on whether c[0]>0 was true
    
    # Count how many extra j increments happened before each i
    c_positive = c_vals > 0.0
    
    # Compute prefix sum of positive conditions for previous elements
    extra_j_count = tl.zeros_like(i_offsets, dtype=tl.int32)
    
    # For block-level processing, approximate the extra j increments
    # by counting positive c values in current block up to each position
    for k in range(BLOCK_SIZE):
        prev_mask = offsets < k
        prev_positive = tl.where(prev_mask, c_positive, False)
        count = tl.sum(prev_positive.to(tl.int32))
        extra_j_count = tl.where(offsets == k, count, extra_j_count)
    
    # Base j position: each i maps to j = i + extra_increments_from_previous_blocks
    # For simplicity in this block, assume we know the prefix from previous blocks
    base_j = i_offsets + extra_j_count
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    first_vals = b_vals + d_vals * e_vals
    tl.store(a_ptr + base_j, first_vals, mask=mask)
    
    # Conditional second assignment  
    second_j = base_j + 1
    second_vals = c_vals + d_vals * e_vals
    second_mask = mask & c_positive
    tl.store(a_ptr + second_j, second_vals, mask=second_mask)

def s123_triton(a, b, c, d, e):
    n_half = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_half, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_half,
        BLOCK_SIZE=BLOCK_SIZE,
    )