import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load block of a values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # For each element in this block
    for local_i in range(BLOCK_SIZE):
        global_i = block_start + local_i
        
        # Skip if out of bounds
        skip = global_i >= n
        
        a_val = tl.load(a_ptr + global_i, mask=~skip)
        
        # Only process if element is in bounds and positive
        process = (global_i < n) & (a_val > 0.0)
        
        if process:
            # Count positive elements up to and including index global_i
            j = -1
            for k in range(global_i + 1):
                skip_k = k >= n
                prev_a_val = tl.load(a_ptr + k, mask=~skip_k)
                is_positive = (prev_a_val > 0.0) & (~skip_k)
                if is_positive:
                    j += 1
            
            # Load b[j] and store to a[global_i]
            if j >= 0:
                b_val = tl.load(b_ptr + j)
                tl.store(a_ptr + global_i, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=BLOCK_SIZE
    )