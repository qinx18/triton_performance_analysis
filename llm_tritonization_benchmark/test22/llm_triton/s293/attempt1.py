import triton
import triton.language as tl
import torch

@triton.jit
def s293_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load the original a[0] value from the read-only copy
    a0_val = tl.load(a_copy_ptr)
    
    # Store a[0] value to all positions
    tl.store(a_ptr + indices, a0_val, mask=mask)

def s293_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s293_kernel[grid](a, a_copy, n, BLOCK_SIZE=BLOCK_SIZE)