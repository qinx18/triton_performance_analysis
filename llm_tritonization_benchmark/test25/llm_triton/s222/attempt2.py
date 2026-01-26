import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n):
    # This kernel must run with exactly one thread due to strict sequential dependency
    # Each thread processes the entire array sequentially
    thread_id = tl.program_id(0)
    
    if thread_id != 0:
        return
    
    # Process sequentially from i=1 to n-1
    for i in range(1, n):
        # Load values for a[i] += b[i] * c[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Update a[i]
        a_val = a_val + b_val * c_val
        tl.store(a_ptr + i, a_val)
        
        # Load e[i-1] and compute e[i] = e[i-1] * e[i-1]
        e_prev = tl.load(e_ptr + i - 1)
        e_new = e_prev * e_prev
        tl.store(e_ptr + i, e_new)
        
        # Update a[i] again: a[i] -= b[i] * c[i] (use original a[i] value)
        a_orig = tl.load(a_ptr + i)
        a_final = a_orig - b_val * c_val
        tl.store(a_ptr + i, a_final)

def s222_triton(a, b, c, e):
    n = a.shape[0]
    
    # Launch with single thread since computation is strictly sequential
    grid = (1,)
    s222_kernel[grid](a, b, c, e, n)