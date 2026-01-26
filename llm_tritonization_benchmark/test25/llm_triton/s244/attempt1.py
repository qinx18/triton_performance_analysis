import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)
    
    # Epilogue - execute S2 only for last iteration (i = n_elements - 2)
    if n_elements >= 2:
        last_i = n_elements - 2
        if tl.program_id(0) == 0:  # Only first block executes epilogue
            # Load required values for last iteration
            b_val = tl.load(b_ptr + last_i)
            a_next_val = tl.load(a_ptr + last_i + 1)
            d_val = tl.load(d_ptr + last_i)
            
            # S2: a[i+1] = b[i] + a[i+1] * d[i]
            result = b_val + a_next_val * d_val
            tl.store(a_ptr + last_i + 1, result)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )