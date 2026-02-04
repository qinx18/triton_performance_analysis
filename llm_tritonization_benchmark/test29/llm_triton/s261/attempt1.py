import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get thread block offsets
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Prologue: compute a[1] using c[0]
    if block_start == 0:
        # Load values for prologue
        a_val = tl.load(a_ptr + 1)
        b_val = tl.load(b_ptr + 1)
        c_val = tl.load(c_ptr + 0)
        
        # Compute a[1] = (a[1] + b[1]) + c[0]
        result = (a_val + b_val) + c_val
        tl.store(a_ptr + 1, result)
    
    # Main parallel loop: each thread handles one i value from 1 to n-2
    current_offsets = block_start + offsets
    mask = (current_offsets >= 1) & (current_offsets < n - 1)
    
    # Step 1: compute c[i] = c[i] * d[i]
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    c_results = c_vals * d_vals
    tl.store(c_ptr + current_offsets, c_results, mask=mask)
    
    # Step 2: compute a[i+1] = (a[i+1] + b[i+1]) + c[i]
    next_offsets = current_offsets + 1
    next_mask = (current_offsets >= 1) & (current_offsets < n - 1)
    
    a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
    b_next_vals = tl.load(b_ptr + next_offsets, mask=next_mask)
    a_results = (a_next_vals + b_next_vals) + c_results
    tl.store(a_ptr + next_offsets, a_results, mask=next_mask)
    
    # Epilogue: compute c[n-1] = c[n-1] * d[n-1]
    if block_start + BLOCK_SIZE >= n - 1:
        epilogue_idx = n - 1
        if epilogue_idx >= 1:
            c_val = tl.load(c_ptr + epilogue_idx)
            d_val = tl.load(d_ptr + epilogue_idx)
            result = c_val * d_val
            tl.store(c_ptr + epilogue_idx, result)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size for main parallel loop (i from 1 to n-2)
    grid_size = triton.cdiv(n - 1, BLOCK_SIZE)
    
    s261_kernel[(grid_size,)](
        a, b, c, d, n, BLOCK_SIZE
    )
    
    return a, c