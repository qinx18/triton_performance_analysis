import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(
    a_ptr, b_ptr, c_ptr, e_ptr, e_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # First operation: a[i] += b[i] * c[i]
    bc_product = b_vals * c_vals
    a_vals = a_vals + bc_product
    
    # Store intermediate result
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Second operation: e[i] = e[i-1] * e[i-1]
    # Load e[i-1] from the read-only copy
    prev_offsets = offsets - 1
    prev_mask = (prev_offsets >= 0) & mask
    e_prev_vals = tl.load(e_copy_ptr + prev_offsets, mask=prev_mask)
    
    # Compute e[i] = e[i-1] * e[i-1]
    e_new_vals = e_prev_vals * e_prev_vals
    
    # Store new e values
    tl.store(e_ptr + offsets, e_new_vals, mask=mask)
    
    # Third operation: a[i] -= b[i] * c[i]
    a_vals = a_vals - bc_product
    
    # Store final a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of e for WAR race condition handling
    e_copy = e.clone()
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements - 1, meta['BLOCK_SIZE']),)
    
    s222_kernel[grid](
        a, b, c, e, e_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )