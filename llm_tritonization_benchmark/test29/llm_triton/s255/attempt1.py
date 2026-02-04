import triton
import triton.language as tl
import torch

@triton.jit
def s255_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute x values (previous b value)
    x_vals = tl.where(offsets == 0, 
                      tl.load(b_ptr + (n_elements - 1)),  # b[LEN_1D-1] for i=0
                      tl.where(mask, 
                               tl.load(b_ptr + offsets - 1, mask=offsets > 0), 
                               0.0))  # b[i-1] for i>0
    
    # Compute y values (two iterations back)
    y_vals = tl.where(offsets == 0,
                      tl.load(b_ptr + (n_elements - 2)),  # b[LEN_1D-2] for i=0
                      tl.where(offsets == 1,
                               tl.load(b_ptr + (n_elements - 1)),  # b[LEN_1D-1] for i=1
                               tl.where(mask,
                                        tl.load(b_ptr + offsets - 2, mask=offsets > 1),
                                        0.0)))  # b[i-2] for i>1
    
    # Compute a[i] = (b[i] + x + y) * 0.333
    result = (b_vals + x_vals + y_vals) * 0.333
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s255_triton(a, b, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s255_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )