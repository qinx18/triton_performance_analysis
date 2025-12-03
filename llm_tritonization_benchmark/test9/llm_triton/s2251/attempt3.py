import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process elements sequentially within this block
    s = 0.0
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load single values
            e_val = tl.load(e_ptr + idx)
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            
            # a[i] = s * e[i]
            a_val = s * e_val
            
            # s = b[i] + c[i]
            s = b_val + c_val
            
            # b[i] = a[i] + d[i]
            b_val = a_val + d_val
            
            # Store results
            tl.store(a_ptr + idx, a_val)
            tl.store(b_ptr + idx, b_val)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b