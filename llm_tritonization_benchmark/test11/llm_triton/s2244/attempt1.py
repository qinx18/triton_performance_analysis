import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop - execute all iterations for S1 (a[i] = b[i] + c[i])
    for block_start in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < (n_elements - 1)
        
        # S1: a[i] = b[i] + c[i] - execute for all iterations
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        result_s1 = b_vals + c_vals
        tl.store(a_ptr + current_offsets, result_s1, mask=mask)
    
    # Epilogue - execute S0 only for last iteration (i = N-2)
    last_i = n_elements - 2
    if last_i >= 0:
        # S0: a[i+1] = b[i] + e[i] at i = N-2
        b_val = tl.load(b_ptr + last_i)
        e_val = tl.load(e_ptr + last_i)
        result_s0 = b_val + e_val
        tl.store(a_ptr + last_i + 1, result_s0)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    
    s2244_kernel[(1,)](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )