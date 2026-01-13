import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i] (using original b[i])
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)
    
    # Epilogue - execute S2 only for the last iteration (i = n_elements-1)
    last_i = n_elements - 1
    if last_i >= 0:
        # Load b[last_i], a[last_i+1], d[last_i]
        b_val = tl.load(b_ptr + last_i)
        a_val = tl.load(a_copy_ptr + last_i + 1)
        d_val = tl.load(d_ptr + last_i)
        
        # S2: a[last_i+1] = b[last_i] + a[last_i+1] * d[last_i]
        result = b_val + a_val * d_val
        tl.store(a_ptr + last_i + 1, result)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s244_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )