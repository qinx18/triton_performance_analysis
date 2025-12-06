import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed with a single block due to the sequential dependency
    # Each element depends on the previous computation of s
    
    s = 0.0
    
    # Process elements sequentially in chunks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        elements_in_block = min(BLOCK_SIZE, n_elements - block_start)
        
        # Process each element in the current block sequentially
        for idx in range(elements_in_block):
            current_i = block_start + idx
            
            # Load scalar values
            e_val = tl.load(e_ptr + current_i)
            b_val = tl.load(b_ptr + current_i)
            c_val = tl.load(c_ptr + current_i)
            d_val = tl.load(d_ptr + current_i)
            
            # a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + current_i, a_val)
            
            # s = b[i] + c[i]
            s = b_val + c_val
            
            # b[i] = a[i] + d[i]
            b_new = a_val + d_val
            tl.store(b_ptr + current_i, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Use single block execution due to sequential dependencies
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )