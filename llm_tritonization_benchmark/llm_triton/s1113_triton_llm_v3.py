import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    len_1d,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read from copy for a[LEN_1D/2]
    mid_idx = len_1d // 2
    a_mid_val = tl.load(a_copy_ptr + mid_idx)
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute result
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.numel()
    len_1d = n_elements
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Launch parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a, a_copy, b,
        n_elements, len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )