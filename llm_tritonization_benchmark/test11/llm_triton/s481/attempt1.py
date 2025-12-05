import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Check if any d[i] < 0 - in GPU context, we skip negative values instead of exit
    valid_mask = mask & (d_vals >= 0.0)
    
    # Load other arrays only where valid
    a_vals = tl.load(a_ptr + idx, mask=valid_mask)
    b_vals = tl.load(b_ptr + idx, mask=valid_mask)
    c_vals = tl.load(c_ptr + idx, mask=valid_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Check if any element in d is negative - if so, return early to simulate exit behavior
    if torch.any(d < 0.0):
        return
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )