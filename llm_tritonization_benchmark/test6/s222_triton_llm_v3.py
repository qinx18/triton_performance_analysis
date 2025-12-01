import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially due to recurrence in e[i] = e[i-1] * e[i-1]
    for i in range(1, n_elements):
        if i >= block_start and i < block_start + BLOCK_SIZE:
            offset = i - block_start
            
            # Load current values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            e_prev = tl.load(e_ptr + i - 1)
            
            # Perform computations
            bc_product = b_val * c_val
            a_val += bc_product
            e_new = e_prev * e_prev
            a_val -= bc_product
            
            # Store results
            tl.store(a_ptr + i, a_val)
            tl.store(e_ptr + i, e_new)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block due to sequential dependency
    grid = (1,)
    s222_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, e