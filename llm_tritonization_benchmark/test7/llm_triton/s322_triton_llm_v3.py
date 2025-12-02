import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially due to dependencies
    for block_start in range(2, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Load previous values (i-1 and i-2)
        prev1_offsets = current_offsets - 1
        prev2_offsets = current_offsets - 2
        
        prev1_mask = (prev1_offsets >= 0) & mask
        prev2_mask = (prev2_offsets >= 0) & mask
        
        a_prev1 = tl.load(a_ptr + prev1_offsets, mask=prev1_mask)
        a_prev2 = tl.load(a_ptr + prev2_offsets, mask=prev2_mask)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_vals + a_prev1 * b_vals + a_prev2 * c_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s322_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Process starting from index 2 due to dependencies
    if n_elements <= 2:
        return
    
    grid = (1,)  # Single block due to sequential dependencies
    
    s322_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )