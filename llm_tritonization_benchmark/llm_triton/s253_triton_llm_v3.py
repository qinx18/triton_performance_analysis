import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute condition: a[i] > b[i]
        condition = a_vals > b_vals
        
        # Compute s = a[i] - b[i] * d[i]
        s_vals = a_vals - b_vals * d_vals
        
        # Apply conditional updates
        # c[i] += s (only where condition is true)
        new_c = tl.where(condition, c_vals + s_vals, c_vals)
        
        # a[i] = s (only where condition is true)
        new_a = tl.where(condition, s_vals, a_vals)
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a, mask=mask)
        tl.store(c_ptr + current_offsets, new_c, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = lambda meta: (1,)
    
    s253_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )