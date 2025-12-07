import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < (n_elements - 1)
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_plus_1_vals = tl.load(a_copy_ptr + idx_plus_1, mask=mask_plus_1)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    a_new = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    tl.store(b_ptr + idx, b_new, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (only for last iteration)
    # Since this is optimization case, we only execute S2 for the last valid index
    is_last = idx == (n_elements - 2)
    last_mask = mask & is_last
    
    if tl.any(last_mask):
        a_plus_1_new = b_new + a_plus_1_vals * d_vals
        tl.store(a_ptr + idx_plus_1, a_plus_1_new, mask=last_mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )