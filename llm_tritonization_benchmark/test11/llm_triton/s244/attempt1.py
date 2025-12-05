import torch
import triton
import triton.language as tl

@triton.jit
def s244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n - 1
        
        # Load values for current iteration
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new_vals = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)
        
        # S2: Only execute at last iteration (i = n-2)
        last_iter_mask = mask & (current_offsets == n - 2)
        if tl.sum(last_iter_mask.to(tl.int32)) > 0:
            a_next_result = b_new_vals + a_next_vals * d_vals
            tl.store(a_ptr + current_offsets + 1, a_next_result, mask=last_iter_mask)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    s244_kernel[(1,)](
        a, a_copy, b, c, d, n, BLOCK_SIZE
    )