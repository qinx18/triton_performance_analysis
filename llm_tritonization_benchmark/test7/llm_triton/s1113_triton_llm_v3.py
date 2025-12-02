import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load from read-only copy and b array
    mid_idx = n_elements // 2
    a_mid_val = tl.load(a_copy_ptr + mid_idx)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE,
    )