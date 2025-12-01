import triton
import triton.language as tl
import torch

@triton.jit
def s151s_kernel(a_ptr, b_ptr, n, inc, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets * inc, mask=mask)
        tl.store(b_ptr + current_offsets, a_vals, mask=mask)

@triton.jit
def s151_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n
    
    # Call s151s equivalent: copy a to b with stride 1
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    tl.store(b_ptr + current_offsets, a_vals, mask=mask)

def s151_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s151_kernel[grid](a, b, n, BLOCK_SIZE)
    
    return b