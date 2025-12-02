import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(block_start, block_start + BLOCK_SIZE):
        if i >= n_elements:
            break
            
        # Load scalar values for current iteration
        mask_i = i < n_elements
        b_val = tl.load(b_ptr + i, mask=mask_i)
        c_val = tl.load(c_ptr + i, mask=mask_i)
        d_val = tl.load(d_ptr + i, mask=mask_i)
        e_val = tl.load(e_ptr + i, mask=mask_i)
        
        # j starts at i (since j = -1 initially, then incremented before first use)
        j = i
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1, mask=mask_i)
        
        # Check condition: if (c[i] > 0.)
        if c_val > 0.0:
            j = j + 1
            # Second assignment: a[j] = c[i] + d[i] * e[i]
            result2 = c_val + d_val * e_val
            mask_j = j < (n_elements * 2)  # Ensure we don't write out of bounds
            tl.store(a_ptr + j, result2, mask=mask_j)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0]  # LEN_1D/2
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )