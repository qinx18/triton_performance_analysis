import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_orig_0, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # All elements get set to a[0] (original value)
    tl.store(a_ptr + indices, tl.broadcast_to(a_orig_0, [BLOCK_SIZE]), mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Handle crossing threshold dependency: save original a[0] before modification
    a_orig_0 = a[0].item()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a,
        a_orig_0,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )