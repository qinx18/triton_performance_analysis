import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    mid_idx,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the value from the middle of the array (read from copy)
    a_mid_val = tl.load(a_copy_ptr + mid_idx)
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute result
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.numel()
    mid_idx = n_elements // 2
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        mid_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )