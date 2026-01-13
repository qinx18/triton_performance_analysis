import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute s values: s = 2.0 * (i + 1) for each element
    s_vals = 2.0 * (offsets + 1).to(tl.float32)
    
    # Compute a values
    a_vals = s_vals * b_vals
    
    # Store a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )