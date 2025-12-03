import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially due to data dependency in b[i] = b[i-1] + ...
    for i in range(1, n_elements):
        # Load single elements
        current_offset = i + tl.arange(0, 1)
        prev_offset = (i - 1) + tl.arange(0, 1)
        
        mask_current = current_offset < n_elements
        mask_prev = prev_offset >= 0
        
        # Load values
        a_i = tl.load(a_ptr + current_offset, mask=mask_current)
        b_prev = tl.load(b_ptr + prev_offset, mask=mask_prev)
        c_i = tl.load(c_ptr + current_offset, mask=mask_current)
        d_i = tl.load(d_ptr + current_offset, mask=mask_current)
        
        # Compute a[i] += c[i] * d[i]
        new_a_i = a_i + c_i * d_i
        tl.store(a_ptr + current_offset, new_a_i, mask=mask_current)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        new_b_i = b_prev + new_a_i + d_i
        tl.store(b_ptr + current_offset, new_b_i, mask=mask_current)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (1,)
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )