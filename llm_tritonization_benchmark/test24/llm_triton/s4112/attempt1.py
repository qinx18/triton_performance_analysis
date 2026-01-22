import triton
import triton.language as tl
import torch

@triton.jit
def s4112_kernel(a_ptr, b_ptr, ip_ptr, s, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < N
    
    # Load indices
    indices = tl.load(ip_ptr + i_offsets, mask=mask)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    # Load b[ip[i]] - gather operation
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] += b[ip[i]] * s
    result = a_vals + b_vals * s
    
    # Store result
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s4112_triton(a, b, ip, s):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4112_kernel[grid](
        a, b, ip, s, N, BLOCK_SIZE=BLOCK_SIZE
    )