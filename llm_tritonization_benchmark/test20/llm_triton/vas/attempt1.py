import triton
import triton.language as tl
import torch

@triton.jit
def vas_kernel(a_ptr, b_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Load scatter indices
    scatter_indices = tl.load(ip_ptr + indices, mask=mask)
    
    # Scatter store to a[ip[i]] = b[i]
    tl.store(a_ptr + scatter_indices, b_vals, mask=mask)

def vas_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vas_kernel[grid](a, b, ip, N, BLOCK_SIZE=BLOCK_SIZE)