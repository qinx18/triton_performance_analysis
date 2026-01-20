import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n - 1
    
    # Load b[i] and a[i+1] (which is a[j])
    b_vals = tl.load(b_ptr + idx, mask=mask)
    a_next_vals = tl.load(a_ptr + idx + 1, mask=mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_next_vals + b_vals
    
    # Store result to a[i]
    tl.store(a_ptr + idx, result, mask=mask)

def s121_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)