import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Load c[i+1] values for the second branch
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)  # Allow loading c[i+1]
    c_plus_1_vals = tl.load(c_ptr + idx_plus_1, mask=mask_plus_1, other=0.0)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: b[i] >= 0 -> a[i] = c[i] + d[i] * e[i]
    branch1_result = c_vals + d_vals * e_vals
    
    # Branch 2: b[i] < 0 -> c[i+1] = a[i] + d[i] * d[i]
    branch2_result = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # For branch 1: store to a[i]
    a_result = tl.where(condition, a_vals, branch1_result)  # Keep original a[i] if condition true
    tl.store(a_ptr + idx, a_result, mask=mask)
    
    # For branch 2: store to c[i+1] 
    c_result = tl.where(condition, branch2_result, c_plus_1_vals)  # Store to c[i+1] if condition true
    store_mask_plus_1 = mask_plus_1 & condition
    tl.store(c_ptr + idx_plus_1, c_result, mask=store_mask_plus_1)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c