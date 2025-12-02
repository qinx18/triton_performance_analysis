import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # For c[i+1], we need to load with offset +1 and handle bounds
    c_next_offsets = offsets + 1
    c_next_mask = (offsets < n_elements - 1) & mask
    c_next_vals = tl.load(c_ptr + c_next_offsets, mask=c_next_mask)
    
    # Condition: b[i] < 0
    negative_condition = b_vals < 0.0
    
    # Path 1: b[i] >= 0, execute a[i] = c[i] + d[i] * e[i]
    a_result_path1 = c_vals + d_vals * e_vals
    
    # Path 2: b[i] < 0, execute c[i+1] = a[i] + d[i] * d[i]
    c_result_path2 = a_vals + d_vals * d_vals
    
    # Update a[i] for path 1 (when b[i] >= 0)
    a_final = tl.where(negative_condition, a_vals, a_result_path1)
    tl.store(a_ptr + offsets, a_final, mask=mask)
    
    # Update c[i+1] for path 2 (when b[i] < 0)
    # Only store if we're not at the last element and condition is met
    store_c_mask = c_next_mask & negative_condition
    tl.store(c_ptr + c_next_offsets, c_result_path2, mask=store_c_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )