import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices from ip array
    ip_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Load values from a array
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Gather from b array using indirect indices
    b_vals = tl.load(b_ptr + ip_indices, mask=mask)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store back to a array
    tl.store(a_ptr + indices, result, mask=mask)

def s4112_triton(a, b, ip, s):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s, n_elements, BLOCK_SIZE
    )