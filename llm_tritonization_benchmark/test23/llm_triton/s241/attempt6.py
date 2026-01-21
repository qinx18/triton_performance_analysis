import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Ensure we don't go beyond n-1 (since loop is i < LEN_1D-1)
    mask = offsets < (n - 1)
    
    # Load values for first computation
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    new_a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    
    # Load a[i+1] from the original copy for second computation
    next_offsets = offsets + 1
    next_mask = next_offsets < n
    combined_mask = mask & next_mask
    
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=combined_mask)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    new_b_vals = new_a_vals * a_next_vals * d_vals
    tl.store(b_ptr + offsets, new_b_vals, mask=combined_mask)

def s241_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    s241_kernel[grid](a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a, b