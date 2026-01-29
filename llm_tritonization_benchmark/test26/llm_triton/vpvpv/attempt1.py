import triton
import triton.language as tl
import torch

@triton.jit
def vpvpv_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute: a[i] += b[i] + c[i]
        result = a_vals + b_vals + c_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def vpvpv_triton(a, b, c):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    
    vpvpv_kernel[(1,)](
        a, b, c,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a