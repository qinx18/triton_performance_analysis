import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements - 1
        
        # Load b[i-1], c[i], d[i], a[i+1]
        b_prev_vals = tl.load(b_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        a_next_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
        
        # Compute a[i] = b[i-1] + c[i]
        a_vals = b_prev_vals + c_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # Compute b[i] = a[i+1] * d[i]
        b_vals = a_next_vals * d_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy of array a to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 1024
    
    s1213_kernel[(1,)](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE
    )