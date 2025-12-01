import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    ip_indices = tl.load(ip_ptr + offsets, mask=mask)
    
    # Load b values using gathered indices
    b_values = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Load a values
    a_values = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute: a[i] += b[ip[i]] * s
    result = a_values + b_values * s
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )