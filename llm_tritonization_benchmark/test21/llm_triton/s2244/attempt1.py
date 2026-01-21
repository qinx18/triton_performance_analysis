import triton
import triton.language as tl
import torch

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Main loop for all iterations except the last one
    for i in range(block_start, min(block_start + BLOCK_SIZE, n - 1)):
        if i >= n - 1:
            break
            
        # Load values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # S1: a[i] = b[i] + c[i] (executed for all iterations)
        tl.store(a_ptr + i, b_val + c_val)
    
    # Epilogue - handle the last iteration separately
    # S0: a[i+1] = b[i] + e[i] for i = n-2 (last iteration)
    if pid == 0:  # Only first block handles this
        last_i = n - 2
        if last_i >= 0:
            b_val = tl.load(b_ptr + last_i)
            e_val = tl.load(e_ptr + last_i)
            tl.store(a_ptr + last_i + 1, b_val + e_val)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2244_kernel[grid](a, b, c, e, n, BLOCK_SIZE=BLOCK_SIZE)