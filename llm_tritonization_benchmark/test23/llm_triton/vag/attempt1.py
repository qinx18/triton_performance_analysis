import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < N
    
    # Load indices
    ip_offsets = block_start + offsets
    indices = tl.load(ip_ptr + ip_offsets, mask=mask)
    
    # Gather from b using indices
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Store to a
    a_offsets = block_start + offsets
    tl.store(a_ptr + a_offsets, b_vals, mask=mask)

def vag_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip,
        N, BLOCK_SIZE
    )