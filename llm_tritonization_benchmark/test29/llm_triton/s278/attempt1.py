import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = tl.load(c_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: a[i] > 0
    condition = a > 0.0
    
    # Compute both branches
    b_updated = -b + d * e  # for a[i] <= 0
    c_updated = -c + d * e  # for a[i] > 0
    
    # Select based on condition
    b_result = tl.where(condition, b, b_updated)
    c_result = tl.where(condition, c_updated, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_result = b_result + c_result * d
    
    # Store results
    tl.store(a_ptr + idx, a_result, mask=mask)
    tl.store(b_ptr + idx, b_result, mask=mask)
    tl.store(c_ptr + idx, c_result, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )