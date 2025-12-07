import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Main loop computation: a[i] = b[i] + c[i] for i < n-1
    mask = current_offsets < (n_elements - 1)
    
    # Load inputs
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = b[i] + c[i]
    val_for_curr = b_vals + c_vals
    
    # Store a[i] = b[i] + c[i]
    tl.store(a_ptr + current_offsets, val_for_curr, mask=mask)

@triton.jit
def s2244_epilogue_kernel(a_ptr, b_ptr, e_ptr, n_elements):
    # Handle the last iteration: a[n-1] = b[n-2] + e[n-2]
    if n_elements >= 2:
        last_idx = n_elements - 2
        b_val = tl.load(b_ptr + last_idx)
        e_val = tl.load(e_ptr + last_idx)
        result = b_val + e_val
        tl.store(a_ptr + (last_idx + 1), result)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    if n_elements < 2:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Main computation
    s2244_kernel[grid](
        a, b, c, e, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Epilogue for the last iteration
    s2244_epilogue_kernel[(1,)](a, b, e, n_elements)