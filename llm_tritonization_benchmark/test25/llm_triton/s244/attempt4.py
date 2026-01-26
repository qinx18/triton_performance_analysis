import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
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
        
        # Load updated b values for S2
        updated_b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        # S2: a[i+1] = b[i] + a[i+1] * d[i] (with bounds check)
        next_offsets = current_offsets + 1
        next_mask = mask & (next_offsets < n_elements)
        
        if tl.max(next_mask.to(tl.int32)) > 0:
            a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
            result = updated_b_vals + a_next_vals * d_vals
            tl.store(a_ptr + next_offsets, result, mask=next_mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s244_kernel[grid](
        a, b, c, d, n_elements, BLOCK_SIZE
    )