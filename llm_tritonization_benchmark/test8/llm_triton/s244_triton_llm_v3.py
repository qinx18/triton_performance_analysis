import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process the main loop sequentially within each block
    for i in range(block_start, min(block_start + BLOCK_SIZE, n_elements - 1)):
        # Load single elements for sequential processing
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        a_i_plus_1 = tl.load(a_ptr + i + 1)
        
        # First statement: a[i] = b[i] + c[i] * d[i]
        a_i = b_i + c_i * d_i
        tl.store(a_ptr + i, a_i)
        
        # Second statement: b[i] = c[i] + b[i]
        b_i_new = c_i + b_i
        tl.store(b_ptr + i, b_i_new)
        
        # Third statement: a[i+1] = b[i] + a[i+1] * d[i]
        a_i_plus_1_new = b_i_new + a_i_plus_1 * d_i
        tl.store(a_ptr + i + 1, a_i_plus_1_new)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )