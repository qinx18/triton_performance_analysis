import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block of indices
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process all iterations from 0 to n-2 (since loop goes to n-1 exclusive)
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_ptr + current_offsets + 1, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_new = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        # Only execute for the last iteration (i = n-2) to avoid overwrite
        last_iter_mask = mask & (current_offsets == (n - 2))
        if tl.sum(last_iter_mask.to(tl.int32)) > 0:
            a_final = b_new + a_next_vals * d_vals
            tl.store(a_ptr + current_offsets + 1, a_final, mask=last_iter_mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s244_kernel[(1,)](
        a, b, c, d, n, BLOCK_SIZE
    )