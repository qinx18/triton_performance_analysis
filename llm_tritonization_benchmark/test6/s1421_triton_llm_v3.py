import triton
import triton.language as tl
import torch

@triton.jit
def s1421_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load from a[i]
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Load from xx[i] which is b[n_elements + i] (xx points to &b[LEN_1D/2])
    xx_vals = tl.load(b_ptr + n_elements + block_start + offsets, mask=mask)
    
    # Compute b[i] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to b[i]
    tl.store(b_ptr + block_start + offsets, result, mask=mask)

def s1421_triton(a, b):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )