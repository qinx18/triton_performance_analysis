import triton
import triton.language as tl
import torch

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
    
    # Load a[LEN_1D/2] from copy (read-only)
    a_mid = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[LEN_1D/2] + b[i]
    result = a_mid + b_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.shape[0]
    mid_idx = n_elements // 2
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        mid_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )