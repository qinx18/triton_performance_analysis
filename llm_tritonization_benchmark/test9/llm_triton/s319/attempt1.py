import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    local_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = c[i] + d[i]
        a_vals = c_vals + d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # sum += a[i]
        local_sum += tl.sum(tl.where(mask, a_vals, 0.0))
        
        # Compute b[i] = c[i] + e[i]
        b_vals = c_vals + e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # sum += b[i]
        local_sum += tl.sum(tl.where(mask, b_vals, 0.0))
    
    return local_sum

def s319_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to handle sequential reduction
    grid = (1,)
    
    result = s319_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result