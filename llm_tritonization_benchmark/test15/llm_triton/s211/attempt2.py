import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n):
    # First iteration (prologue)
    if n > 1:
        b_prev = tl.load(b_copy_ptr + 0)
        c_val = tl.load(c_ptr + 1)
        d_val = tl.load(d_ptr + 1)
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + 1, a_val)
    
    # Main loop with reordered statements
    for i in range(1, n - 2):
        # Producer: compute b[i]
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)
        
        # Consumer: compute a[i+1] using just computed b[i]
        b_current = tl.load(b_ptr + i)
        c_next = tl.load(c_ptr + i + 1)
        d_next = tl.load(d_ptr + i + 1)
        a_next = b_current + c_next * d_next
        tl.store(a_ptr + i + 1, a_next)
    
    # Last iteration (epilogue)
    if n > 2:
        i = n - 2
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    b_copy = b.clone()
    
    # Launch with single thread for reordered computation
    grid = (1,)
    s211_kernel[grid](a, b, b_copy, c, d, e, n)