import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, b_copy_ptr, n_elements):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Prologue: handle first element
    if n_elements > 2:
        b_0 = tl.load(b_copy_ptr + 0)
        c_1 = tl.load(c_ptr + 1)
        d_1 = tl.load(d_ptr + 1)
        a_1 = b_0 + c_1 * d_1
        tl.store(a_ptr + 1, a_1)
    
    # Main loop with statement reordering
    for i in range(1, n_elements - 2):
        # Producer first: compute b[i]
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)
        
        # Consumer second: compute a[i+1] using just computed b[i]
        c_next = tl.load(c_ptr + i + 1)
        d_next = tl.load(d_ptr + i + 1)
        a_next = b_val + c_next * d_next
        tl.store(a_ptr + i + 1, a_next)
    
    # Epilogue: handle last producer
    if n_elements > 2:
        i_last = n_elements - 2
        b_last = tl.load(b_copy_ptr + i_last + 1)
        e_last = tl.load(e_ptr + i_last)
        d_last = tl.load(d_ptr + i_last)
        b_val_last = b_last - e_last * d_last
        tl.store(b_ptr + i_last, b_val_last)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    grid = (1,)
    
    s211_kernel[grid](
        a, b, c, d, e, b_copy, n_elements
    )