import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially due to data dependencies
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Calculate k indices: k = 2*i + 1 (since j starts at -1 and increments by 2)
        k_offsets = 2 * current_offsets + 1
        k_mask = k_offsets < (2 * n_elements)  # Ensure k indices are valid
        
        # Load required data
        b_vals = tl.load(b_ptr + k_offsets, mask=mask & k_mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + k_offsets, mask=mask & k_mask, other=0.0)
        
        # Compute a[i] = b[k] - d[i]
        a_vals = b_vals - d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Compute b[k] = a[i] + c[k]
        b_new_vals = a_vals + c_vals
        tl.store(b_ptr + k_offsets, b_new_vals, mask=mask & k_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b