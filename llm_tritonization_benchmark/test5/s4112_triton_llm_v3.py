import torch
import triton
import triton.language as tl

@triton.jit
def s4112_kernel(
    a_ptr, b_ptr, ip_ptr, s,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    ip_offsets = block_start + offsets
    indices = tl.load(ip_ptr + ip_offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + ip_offsets, mask=mask, other=0.0)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result
    tl.store(a_ptr + ip_offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )