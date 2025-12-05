import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements sequentially due to b[i] = b[i-1] dependency
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = strip_start + offsets
        mask = current_offsets < n_elements
        
        # a[i] += c[i] * d[i]
        c_vals = tl.load(c_ptr + current_offsets + 1, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets + 1, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
        
        new_a_vals = a_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets + 1, new_a_vals, mask=mask)
        
        # b[i] = b[i - 1] + a[i] + d[i]
        b_prev_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        new_b_vals = b_prev_vals + new_a_vals + d_vals
        tl.store(b_ptr + current_offsets + 1, new_b_vals, mask=mask)

def s221_triton(a, b, c, d):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    # Use block size of 1 to handle the sequential dependency properly
    BLOCK_SIZE = 1
    
    # Launch single thread block to process sequentially
    s221_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )