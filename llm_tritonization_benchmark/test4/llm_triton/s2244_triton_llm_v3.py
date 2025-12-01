import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load b[i] + e[i] for a[i+1] = b[i] + e[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        result1 = b_vals + e_vals
        
        # Load b[i] + c[i] for a[i] = b[i] + c[i]
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        result2 = b_vals + c_vals
        
        # Store a[i+1] = b[i] + e[i]
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < (n_elements + 1))
        tl.store(a_ptr + next_offsets, result1, mask=next_mask)
        
        # Store a[i] = b[i] + c[i]
        tl.store(a_ptr + current_offsets, result2, mask=mask)

def s2244_triton(a, b, c, e):
    n_elements = len(b) - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a