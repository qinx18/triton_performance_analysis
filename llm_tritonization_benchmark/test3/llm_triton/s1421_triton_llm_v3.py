import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from a[i] and xx[i] (which is b[LEN_1D/2 + i])
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    xx_vals = tl.load(b_ptr + n_elements + offsets, mask=mask)
    
    # Compute b[i] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store result to b[i]
    tl.store(b_ptr + offsets, result, mask=mask)

def s1421_triton(a, b):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        b, a, n_elements, BLOCK_SIZE
    )
    
    return b