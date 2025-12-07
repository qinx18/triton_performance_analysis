import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Calculate cumulative j indices accounting for conditional increments
    # j starts at 0 for i=0, then increments by 1 for each i, plus 1 more if c[i] > 0
    j_base = block_start * 2  # Base j offset for this block
    
    # For each i, calculate how many extra increments happened before this block
    extra_increments = 0
    for prev_block in range(block_id):
        prev_start = prev_block * BLOCK_SIZE
        prev_end = tl.minimum((prev_block + 1) * BLOCK_SIZE, n_elements)
        for prev_i in range(prev_start, prev_end):
            if prev_i < n_elements:
                prev_c = tl.load(c_ptr + prev_i)
                if prev_c > 0.0:
                    extra_increments += 1
    
    # Calculate j offsets for first store
    j_offsets_1 = block_start + i_offsets + extra_increments
    
    # First assignment: a[j] = b[i] + d[i] * e[i]
    result1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_1, result1, mask=mask)
    
    # Second assignment for positive c values
    c_positive = c_vals > 0.0
    combined_mask = mask & c_positive
    j_offsets_2 = j_offsets_1 + 1
    result2 = c_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_2, result2, mask=combined_mask)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 1
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )