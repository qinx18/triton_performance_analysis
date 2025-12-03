import torch
import triton
import triton.language as tl

@triton.jit
def s1113_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[LEN_1D/2] once (scalar broadcast)
    mid_idx = n_elements // 2
    a_mid_val = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute result: a[LEN_1D/2] + b[i]
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a, a_copy, b, n_elements, BLOCK_SIZE
    )