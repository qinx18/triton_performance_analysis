import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Calculate k values for this block: k = 2*i
        k_offsets = 2 * current_offsets
        k_mask = k_offsets < (2 * n_elements)
        
        # Load required data
        b_k = tl.load(b_ptr + k_offsets, mask=k_mask)
        d_i = tl.load(d_ptr + current_offsets, mask=mask)
        c_k = tl.load(c_ptr + k_offsets, mask=k_mask)
        
        # Compute a[i] = b[k] - d[i]
        a_i = b_k - d_i
        
        # Store a[i]
        tl.store(a_ptr + current_offsets, a_i, mask=mask)
        
        # Compute b[k] = a[i] + c[k]
        b_k_new = a_i + c_k
        
        # Store b[k]
        tl.store(b_ptr + k_offsets, b_k_new, mask=k_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s128_kernel[grid](
        a, b, c, d, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )