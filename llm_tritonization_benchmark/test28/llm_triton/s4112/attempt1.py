import triton
import triton.language as tl
import torch

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
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask)
    
    # Load a[i] values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Gather b[ip[i]] values using indirect addressing
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )