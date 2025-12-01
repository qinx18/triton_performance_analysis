import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential processing within each block
    s = 0.0
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        if idx < n_elements:
            # Load scalar values
            e_val = tl.load(e_ptr + idx)
            b_val = tl.load(b_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            
            # Compute a[i] = s * e[i]
            a_val = s * e_val
            tl.store(a_ptr + idx, a_val)
            
            # Update s = b[i] + c[i]
            s = b_val + c_val
            
            # Compute b[i] = a[i] + d[i]
            b_new = a_val + d_val
            tl.store(b_ptr + idx, b_new)

def s2251_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    # Use small block size for sequential dependency
    BLOCK_SIZE = 64
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b