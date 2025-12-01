import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in blocks
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load c[i+1] for the second branch (shift by 1)
        c_next_offsets = current_offsets + 1
        c_next_mask = (c_next_offsets < (n_elements + 1)) & mask
        c_next_vals = tl.load(c_ptr + c_next_offsets, mask=c_next_mask, other=0.0)
        
        # Condition: b[i] < 0
        condition = b_vals < 0.0
        
        # First branch: a[i] = c[i] + d[i] * e[i]
        branch1_result = c_vals + d_vals * e_vals
        
        # Second branch: c[i+1] = a[i] + d[i] * d[i]
        branch2_result = a_vals + d_vals * d_vals
        
        # Store results based on condition
        # For a[i]: store branch1_result when b[i] >= 0 (condition is false)
        a_store_mask = mask & ~condition
        tl.store(a_ptr + current_offsets, branch1_result, mask=a_store_mask)
        
        # For c[i+1]: store branch2_result when b[i] < 0 (condition is true)
        c_store_mask = c_next_mask & condition
        tl.store(c_ptr + c_next_offsets, branch2_result, mask=c_store_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )