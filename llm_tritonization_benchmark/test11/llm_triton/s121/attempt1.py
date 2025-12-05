import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices for this block
    i_indices = block_start + offsets
    j_indices = i_indices + 1
    
    # Mask for valid elements (i < n_elements-1)
    mask_i = i_indices < n_elements - 1
    mask_j = j_indices < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + i_indices, mask=mask_i, other=0.0)
    a_j_vals = tl.load(a_ptr + j_indices, mask=mask_j, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_indices, result, mask=mask_i)

def s121_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a