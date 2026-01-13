import triton
import triton.language as tl
import torch

@triton.jit
def s4117_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load b[i] and d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute c[i/2] indices
    c_indices = indices // 2
    c_mask = (indices < N) & (c_indices < N)
    c_vals = tl.load(c_ptr + c_indices, mask=c_mask, other=0.0)
    
    # Compute a[i] = b[i] + c[i/2] * d[i]
    result = b_vals + c_vals * d_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s4117_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s4117_kernel[grid](a, b, c, d, N, BLOCK_SIZE)
    
    return a