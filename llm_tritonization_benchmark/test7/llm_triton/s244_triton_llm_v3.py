import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially to handle dependencies
    for i in range(block_start, min(block_start + BLOCK_SIZE, n_elements - 1)):
        # Load current values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_next = tl.load(a_ptr + i + 1)
        
        # First computation: a[i] = b[i] + c[i] * d[i]
        a_val = b_val + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Second computation: b[i] = c[i] + b[i]
        b_new = c_val + b_val
        tl.store(b_ptr + i, b_new)
        
        # Third computation: a[i+1] = b[i] + a[i+1] * d[i]
        a_next_new = b_new + a_next * d_val
        tl.store(a_ptr + i + 1, a_next_new)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch kernel with sequential processing due to dependencies
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )