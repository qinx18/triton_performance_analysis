import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Process elements sequentially within each block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= n_elements:
            break
            
        # Load single elements
        b_val = tl.load(b_ptr + idx)
        c_val = tl.load(c_ptr + idx)
        a_val = tl.load(a_ptr + idx)
        
        # Update a[i]
        a_val = a_val + b_val * c_val
        tl.store(a_ptr + idx, a_val)
        
        # Check break condition
        if c_val > b_val:
            break

def s482_triton(a, b, c):
    n_elements = a.numel()
    
    # Use smaller block size to better handle early exit condition
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )