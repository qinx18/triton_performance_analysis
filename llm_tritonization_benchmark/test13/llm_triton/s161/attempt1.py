import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    strip_id = tl.program_id(0)
    strip_start = strip_id
    
    if strip_start >= n_elements:
        return
    
    # Each strip processes exactly 1 element
    offsets = tl.arange(0, 1)
    idx = strip_start + offsets
    
    # Load values
    mask = idx < n_elements
    b_val = tl.load(b_ptr + idx, mask=mask)
    a_val = tl.load(a_ptr + idx, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Conditional logic: if (b[i] < 0.0)
    condition = b_val < 0.0
    
    # Path 1: a[i] = c[i] + d[i] * e[i]
    result_a = c_val + d_val * e_val
    
    # Path 2: c[i+1] = a[i] + d[i] * d[i]
    result_c = a_val + d_val * d_val
    
    # Store results based on condition
    # Always update a[i] when condition is False
    tl.store(a_ptr + idx, tl.where(condition, a_val, result_a), mask=mask)
    
    # Update c[i+1] when condition is True (and i+1 is valid)
    next_idx = idx + 1
    next_mask = (next_idx < (n_elements + 1)) & condition & mask
    if tl.sum(next_mask.to(tl.int32)) > 0:
        tl.store(c_ptr + next_idx, result_c, mask=next_mask)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Must process strips sequentially due to RAW dependency
    BLOCK_SIZE = 1
    
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        grid = (1,)
        s161_kernel[grid](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )