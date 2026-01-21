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
    indices = block_start + offsets
    
    # Main loop - S1: a[i] = b[i] + c[i] for all iterations
    mask = (indices < n - 1)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    tl.store(a_ptr + indices, b_vals + c_vals, mask=mask)
    
    # Epilogue - S0: a[i+1] = b[i] + e[i] only for last iteration (i = n-2)
    last_i = n - 2
    epilogue_mask = (indices == last_i) & (last_i >= 0)
    b_val_epilogue = tl.load(b_ptr + indices, mask=epilogue_mask)
    e_val_epilogue = tl.load(e_ptr + indices, mask=epilogue_mask)
    tl.store(a_ptr + indices + 1, b_val_epilogue + e_val_epilogue, mask=epilogue_mask)

def s2244_triton(a, b, c, e):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2244_kernel[grid](a, b, c, e, n, BLOCK_SIZE=BLOCK_SIZE)