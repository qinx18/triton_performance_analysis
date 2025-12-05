import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program processes one sequential strip
    strip_id = tl.program_id(0)
    strip_start = strip_id * 1
    
    # Check if this strip is valid
    if strip_start >= n_elements:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = idx < n_elements
    
    # Load b[i] to check condition
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    
    # Load other required values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Load c[i+1] values for the second branch (need to check bounds)
    c_next_mask = (idx + 1) < (n_elements + 1)  # c array has LEN_1D elements
    c_next_vals = tl.load(c_ptr + idx + 1, mask=mask & c_next_mask, other=0.0)
    
    # Condition: b[i] < 0
    condition = b_vals < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i] (when b[i] >= 0)
    branch1_result = c_vals + d_vals * e_vals
    
    # Branch 2: c[i+1] = a[i] + d[i] * d[i] (when b[i] < 0)
    branch2_result = a_vals + d_vals * d_vals
    
    # Store results for branch 1 (a[i])
    branch1_mask = mask & (~condition)
    tl.store(a_ptr + idx, branch1_result, mask=branch1_mask)
    
    # Store results for branch 2 (c[i+1])
    branch2_mask = mask & condition & c_next_mask
    tl.store(c_ptr + idx + 1, branch2_result, mask=branch2_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 1  # Must be 1 due to RAW dependency
    num_strips = n_elements  # Each strip has size 1
    
    # Process strips sequentially due to c[i+1] = ... c[i] dependency
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        remaining = min(BLOCK_SIZE, n_elements - strip_start)
        if remaining <= 0:
            break
            
        # Launch kernel for this single strip
        s161_kernel[(1,)](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )