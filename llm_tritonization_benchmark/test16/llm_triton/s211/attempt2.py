import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, N):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Prologue: First consumer uses original b[0]
    b_0 = tl.load(b_copy_ptr + 0)
    c_1 = tl.load(c_ptr + 1)
    d_1 = tl.load(d_ptr + 1)
    a_1 = b_0 + c_1 * d_1
    tl.store(a_ptr + 1, a_1)
    
    # Main loop: Producer first, then shifted consumer
    for i in range(1, N-2):
        # Producer: b[i] = b[i+1] - e[i] * d[i]
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)
        
        # Consumer shifted: a[i+1] = b[i] + c[i+1] * d[i+1]
        c_next = tl.load(c_ptr + i + 1)
        d_next = tl.load(d_ptr + i + 1)
        a_val = b_val + c_next * d_next
        tl.store(a_ptr + i + 1, a_val)
    
    # Epilogue: Last producer has no consumer
    if N >= 3:
        i = N - 2
        b_last = tl.load(b_copy_ptr + i + 1)
        e_last = tl.load(e_ptr + i)
        d_last = tl.load(d_ptr + i)
        b_final = b_last - e_last * d_last
        tl.store(b_ptr + i, b_final)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    # Launch single thread for reordered computation
    grid = (1,)
    s211_kernel[grid](
        a, b, b_copy, c, d, e, N
    )