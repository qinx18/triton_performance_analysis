import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load from a and xx (which is b[n:2*n])
    a_vals = tl.load(a_ptr + indices, mask=mask)
    xx_vals = tl.load(b_ptr + n + indices, mask=mask)
    
    # Compute b[i] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to b[0:n]
    tl.store(b_ptr + indices, result, mask=mask)

def s1421_triton(a, b):
    n = a.shape[0] // 2
    
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    s1421_kernel[grid](
        a, b, n,
        BLOCK_SIZE=256
    )