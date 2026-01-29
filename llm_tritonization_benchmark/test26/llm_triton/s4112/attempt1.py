import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
    
    # Load a values
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals * s
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s4112_triton(a, b, ip):
    N = a.shape[0]
    s = 1.0  # Default scalar value
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s, N,
        BLOCK_SIZE=BLOCK_SIZE
    )