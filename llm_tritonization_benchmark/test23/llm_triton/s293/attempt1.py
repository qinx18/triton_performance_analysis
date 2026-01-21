import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load a[0] from the read-only copy
    a_0_val = tl.load(a_copy_ptr)
    
    # Store a[0] to all elements a[i] = a[0]
    tl.store(a_ptr + offsets, a_0_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s293_kernel[grid](
        a, a_copy, n,
        BLOCK_SIZE=BLOCK_SIZE
    )