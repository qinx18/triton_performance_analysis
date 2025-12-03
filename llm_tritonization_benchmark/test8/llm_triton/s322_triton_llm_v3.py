import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially due to dependencies
    for i in range(2, n_elements):
        # Load current values
        mask_curr = (offsets == 0)
        mask_prev1 = (offsets == 0)
        mask_prev2 = (offsets == 0)
        
        a_curr = tl.load(a_ptr + i + offsets, mask=mask_curr)
        a_prev1 = tl.load(a_ptr + (i - 1) + offsets, mask=mask_prev1)
        a_prev2 = tl.load(a_ptr + (i - 2) + offsets, mask=mask_prev2)
        b_curr = tl.load(b_ptr + i + offsets, mask=mask_curr)
        c_curr = tl.load(c_ptr + i + offsets, mask=mask_curr)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
        
        # Store result
        tl.store(a_ptr + i + offsets, result, mask=mask_curr)

def s322_triton(a, b, c):
    n_elements = a.size(0)
    BLOCK_SIZE = 1
    
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )