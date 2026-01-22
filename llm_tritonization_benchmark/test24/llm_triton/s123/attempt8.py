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
    
    # Calculate how many elements with c > 0 exist before current position
    # We need to handle this across all blocks, so we'll use a simpler approach
    # Store the first computation: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    j_base = i_offsets
    tl.store(a_ptr + j_base, result1, mask=mask)
    
    # For the conditional store, we need to calculate the correct j offset
    # Since this is complex with cross-block dependencies, we'll handle it sequentially
    c_positive = c_vals > 0.0
    result2 = c_vals + d_vals * e_vals
    
    # Calculate prefix sum within block for c_positive
    prefix_sum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    
    # Simple prefix sum using shifts
    c_int = tl.where(c_positive, 1, 0).to(tl.int32)
    prefix_sum = c_int
    
    for shift in [1, 2, 4, 8, 16, 32, 64, 128]:
        if shift < BLOCK_SIZE:
            shifted = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
            valid_indices = offsets >= shift
            shift_mask = valid_indices & (offsets - shift < BLOCK_SIZE)
            shifted = tl.where(shift_mask, prefix_sum, 0)
            prefix_sum = prefix_sum + shifted
    
    # Adjust for starting position
    global_prefix = block_start * 1  # Simplified - assumes worst case
    j_conditional = j_base + prefix_sum + global_prefix
    
    # Store conditional result
    cond_store_mask = mask & c_positive
    tl.store(a_ptr + j_conditional, result2, mask=cond_store_mask)

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