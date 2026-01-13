import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Process main loop statements (S0 and S1) for all iterations
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Mask for valid indices (0 to n-2, since loop goes to n-1)
    mask = idx < (n - 1)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_new = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    tl.store(b_ptr + idx, b_new, mask=mask)

@triton.jit
def s244_epilogue_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # Execute S2 only for the last iteration (i = n-2)
    i = n - 2
    if i >= 0:
        # S2: a[i+1] = b[i] + a[i+1] * d[i]
        b_val = tl.load(b_ptr + i)
        a_val = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        a_new = b_val + a_val * d_val
        tl.store(a_ptr + i + 1, a_new)

def s244_triton(a, b, c, d):
    n = a.shape[0]
    
    if n <= 1:
        return
    
    # Main computation for statements S0 and S1
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n - 1, BLOCK_SIZE)
    
    s244_kernel[(grid_size,)](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Epilogue for statement S2 (last iteration only)
    s244_epilogue_kernel[(1,)](a, b, c, d, n)