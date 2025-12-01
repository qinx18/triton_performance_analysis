import torch
import triton
import triton.language as tl

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr, s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Gather from b using indirect indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask, other=0.0)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store back to a
    tl.store(a_ptr + idx, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a