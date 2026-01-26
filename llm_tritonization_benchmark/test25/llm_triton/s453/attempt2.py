import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute s = 2.0 * (i + 1) for each position
    s_vals = 2.0 * (offsets + 1)
    
    # Compute a[i] = s * b[i]
    a_vals = s_vals * b_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s453_triton(a, b):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s453_kernel[grid](
        a, b,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )