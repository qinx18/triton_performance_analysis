import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Prologue: Handle a[1] = b[0] + c[1]
    if pid == 0:
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr + 1)
        tl.store(a_ptr + 1, b_val + c_val)
    
    # Main parallel loop: each thread handles one i from [1, n-2)
    indices = block_start + offsets
    mask = (indices >= 1) & (indices < n - 2)
    
    # Load values for computation
    a_vals = tl.load(a_copy_ptr + indices + 1, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices + 1, mask=mask)
    
    # Compute b[i] = a_copy[i+1] * d[i]
    b_vals = a_vals * d_vals
    tl.store(b_ptr + indices, b_vals, mask=mask)
    
    # Compute a[i+1] = b[i] + c[i+1]
    a_vals_new = b_vals + c_vals
    tl.store(a_ptr + indices + 1, a_vals_new, mask=mask)
    
    # Epilogue: Handle b[n-2] = a_copy[n-1] * d[n-2]
    last_block = (n - 2 + BLOCK_SIZE - 1) // BLOCK_SIZE - 1
    if pid == last_block:
        epilogue_idx = n - 2
        if epilogue_idx >= 1:
            a_val = tl.load(a_copy_ptr + n - 1)
            d_val = tl.load(d_ptr + epilogue_idx)
            tl.store(b_ptr + epilogue_idx, a_val * d_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = ((n - 2 + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s1213_kernel[grid](
        a, b, c, d, a_copy, n, BLOCK_SIZE
    )