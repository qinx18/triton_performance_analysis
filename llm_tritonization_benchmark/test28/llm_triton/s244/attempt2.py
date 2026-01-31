import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (N - 1)
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        
        # For S2: a[i+1] = b[i] + a[i+1] * d[i]
        # Skip S2 for iterations that will be overwritten (all except the last)
        last_iteration_mask = mask & (current_offsets == N - 2)
        
        if tl.sum(last_iteration_mask.to(tl.int32)) > 0:
            # Load a[i+1] values for the last iteration
            a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=last_iteration_mask)
            # Use updated b values for S2
            a_next_new = b_new + a_next_vals * d_vals
            tl.store(a_ptr + current_offsets + 1, a_next_new, mask=last_iteration_mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N - 1, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d, N, BLOCK_SIZE
    )