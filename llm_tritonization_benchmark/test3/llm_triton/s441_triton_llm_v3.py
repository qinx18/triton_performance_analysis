import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    d_val = tl.load(d_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    a_val = tl.load(a_ptr + offsets, mask=mask)
    
    # Conditional computation
    cond1 = d_val < 0.0
    cond2 = d_val == 0.0
    
    # Compute updates for each condition
    update1 = b_val * c_val  # d[i] < 0
    update2 = b_val * b_val  # d[i] == 0
    update3 = c_val * c_val  # d[i] > 0
    
    # Select appropriate update based on conditions
    result = tl.where(cond1, a_val + update1, 
                     tl.where(cond2, a_val + update2, a_val + update3))
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )