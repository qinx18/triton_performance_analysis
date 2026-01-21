import triton
import triton.language as tl
import torch

@triton.jit
def vag_kernel(a_ptr, b_ptr, ip_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < N
    
    # Load indices
    indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Load from b using gathered indices
    b_values = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Store to a
    tl.store(a_ptr + i_offsets, b_values, mask=mask)

def vag_triton(a, b, ip):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vag_kernel[grid](
        a, b, ip, N,
        BLOCK_SIZE=BLOCK_SIZE
    )