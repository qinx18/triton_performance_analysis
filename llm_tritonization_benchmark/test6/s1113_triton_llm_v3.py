import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    LEN_1D,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load the constant value a[LEN_1D/2] from the copy
    mid_idx = LEN_1D // 2
    a_mid_val = tl.load(a_copy_ptr + mid_idx)
    
    # Load b values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Compute result
    result = a_mid_val + b_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.shape[0]
    LEN_1D = n_elements
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1113_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        LEN_1D,
        BLOCK_SIZE,
    )