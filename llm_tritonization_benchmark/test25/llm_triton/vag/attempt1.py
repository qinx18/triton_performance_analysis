import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    indices = tl.load(ip_ptr + block_start + offsets, mask=mask)
    
    # Gather from b using indices
    gathered_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Store to a
    tl.store(a_ptr + block_start + offsets, gathered_vals, mask=mask)

def vag_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vag_kernel[grid](a, b, ip, N, BLOCK_SIZE)