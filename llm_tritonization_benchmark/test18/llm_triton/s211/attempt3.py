import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Single thread processes all elements sequentially
    if tl.program_id(0) != 0:
        return
    
    # Prologue: a[1] = b[0] + c[1] * d[1]
    if n_elements > 2:
        b_0 = tl.load(b_copy_ptr + 0)
        c_1 = tl.load(c_ptr + 1)
        d_1 = tl.load(d_ptr + 1)
        a_1 = b_0 + c_1 * d_1
        tl.store(a_ptr + 1, a_1)
    
    # Main loop: reordered statements
    for i in range(1, n_elements - 2):
        # Producer first: b[i] = b[i+1] - e[i] * d[i]
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)
        
        # Consumer (shifted): a[i+1] = b[i] + c[i+1] * d[i+1]
        c_next = tl.load(c_ptr + i + 1)
        d_next = tl.load(d_ptr + i + 1)
        a_next = b_val + c_next * d_next
        tl.store(a_ptr + i + 1, a_next)
    
    # Epilogue: b[n-2] = b[n-1] - e[n-2] * d[n-2]
    if n_elements > 2:
        i = n_elements - 2
        b_last = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_last - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    # Launch single thread for sequential processing
    grid = (1,)
    s211_kernel[grid](
        a, b, b_copy, c, d, e, n_elements
    )