import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Process all blocks sequentially for reduction
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = c[i] + d[i]
        a_vals = c_vals + d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # sum += a[i]
        sum_val += tl.sum(tl.where(mask, a_vals, 0.0))
        
        # Compute b[i] = c[i] + e[i]
        b_vals = c_vals + e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # sum += b[i]
        sum_val += tl.sum(tl.where(mask, b_vals, 0.0))
    
    return sum_val

def s319_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Single thread to handle the sequential reduction
    grid = (1,)
    
    sum_result = s319_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_result