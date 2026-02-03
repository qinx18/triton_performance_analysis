import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load arrays for current block
        c_vals = tl.load(c_ptr + current_offsets + 1, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets + 1, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Load b values from copy and original
        b_copy_vals_plus1 = tl.load(b_copy_ptr + current_offsets + 1, mask=mask)
        b_copy_vals_minus1 = tl.load(b_copy_ptr + current_offsets - 1, mask=mask & (current_offsets > 0))
        
        # Compute b[i] = b_copy[i + 1] - e[i] * d[i]
        b_new = b_copy_vals_plus1 - e_vals * d_vals
        
        # Compute a[i] = b_copy[i - 1] + c[i] * d[i]
        a_new = b_copy_vals_minus1 + c_vals * d_vals
        
        # Store results
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        tl.store(a_ptr + current_offsets + 1, a_new, mask=mask & (current_offsets + 1 < n_elements))

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    n_elements = N - 2  # Process elements from index 1 to N-2
    
    # Create read-only copy to handle WAR dependencies
    b_copy = b.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )