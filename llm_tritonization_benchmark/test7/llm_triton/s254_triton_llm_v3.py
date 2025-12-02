import torch
import triton
import triton.language as tl

@triton.jit
def s254_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load last element of b as initial x value
    x = tl.load(b_ptr + (n_elements - 1))
    
    # Process elements in blocks
    for i in range(n_elements):
        # Load b[i]
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] = (b[i] + x) * 0.5
        result = (b_val + x) * 0.5
        
        # Store to a[i]
        tl.store(a_ptr + i, result)
        
        # Update x for next iteration
        x = b_val

def s254_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )