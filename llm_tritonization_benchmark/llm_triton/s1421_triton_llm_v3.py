import torch
import triton
import triton.language as tl

@triton.jit
def s1421_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load from a[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Load from xx[i] which is &b[LEN_1D/2] + i = b[LEN_1D/2 + i]
    xx_offsets = offsets + n_elements  # Since xx starts at LEN_1D/2 and we process LEN_1D/2 elements
    xx_vals = tl.load(b_ptr + xx_offsets, mask=mask)
    
    # Compute b[i] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to b[i]
    tl.store(b_ptr + offsets, result, mask=mask)

def s1421_triton(a, b):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )