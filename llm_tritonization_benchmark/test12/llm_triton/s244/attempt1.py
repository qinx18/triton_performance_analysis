import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # S0: a[i] = b[i] + c[i] * d[i]
        a_vals = b_vals + c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # S1: b[i] = c[i] + b[i]
        b_new = c_vals + b_vals
        tl.store(b_ptr + current_offsets, b_new, mask=mask)
    
    # Epilogue - execute S2 only for last iteration (i = N-2)
    i = n_elements - 2
    if i >= 0:
        pid = tl.program_id(0)
        if pid == 0:  # Only one thread executes this
            b_val = tl.load(b_ptr + i)
            a_next_val = tl.load(a_copy_ptr + (i + 1))
            d_val = tl.load(d_ptr + i)
            
            # S2: a[i+1] = b[i] + a[i+1] * d[i]
            a_result = b_val + a_next_val * d_val
            tl.store(a_ptr + (i + 1), a_result)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s244_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )