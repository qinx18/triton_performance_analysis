import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute S0 and S1 for all iterations
    for block_start in range(0, N - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N - 1
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i] - Need to use the NEW b[i] value for S2
        new_b_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] - Use updated b[i] value
        next_offsets = current_offsets + 1
        next_mask = next_offsets < N
        
        # Load a[i+1] values
        a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        a_next_result = new_b_vals + a_next_vals * d_vals
        tl.store(a_ptr + next_offsets, a_next_result, mask=next_mask)

def s244_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s244_kernel[grid](a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE)