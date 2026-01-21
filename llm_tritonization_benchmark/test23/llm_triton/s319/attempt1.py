import triton
import triton.language as tl
import torch

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process all elements in blocks
    total_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load data
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = c[i] + d[i]
        a_vals = c_vals + d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # sum += a[i]
        a_sum = tl.sum(tl.where(mask, a_vals, 0.0))
        total_sum += a_sum
        
        # Compute b[i] = c[i] + e[i]
        b_vals = c_vals + e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # sum += b[i]
        b_sum = tl.sum(tl.where(mask, b_vals, 0.0))
        total_sum += b_sum
    
    return total_sum

def s319_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single thread block to handle reduction properly
    grid = (1,)
    
    result = s319_kernel[grid](
        a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result