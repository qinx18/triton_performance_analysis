import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Check condition: b[i] < 0
    else_mask = b_vals < 0.0
    combined_mask = mask & else_mask
    
    # Compute: c[i+1] = a[i] + d[i] * d[i]
    result = a_vals + d_vals * d_vals
    
    # Store to c[i+1]
    store_idx = idx + 1
    store_mask = combined_mask & (store_idx < (n_elements + 1))
    tl.store(c_ptr + store_idx, result, mask=store_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Check condition: b[i] >= 0
    if_mask = b_vals >= 0.0
    combined_mask = mask & if_mask
    
    # Compute: a[i] = c[i] + d[i] * e[i]
    result = c_vals + d_vals * e_vals
    
    # Store to a[i]
    tl.store(a_ptr + idx, result, mask=combined_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Step 1: Process ALL 'else' branch iterations FIRST
    s161_kernel_else[grid](
        a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Process ALL 'if' branch iterations AFTER step 1 completes
    s161_kernel_if[grid](
        a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE
    )