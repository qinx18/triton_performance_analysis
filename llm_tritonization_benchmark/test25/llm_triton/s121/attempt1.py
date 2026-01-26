import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices
    i_offsets = block_start + offsets
    j_offsets = i_offsets + 1
    
    # Mask for valid elements (i < n_elements-1, so j < n_elements)
    mask_i = i_offsets < (n_elements - 1)
    mask_j = j_offsets < n_elements
    
    # Load a[j] and b[i]
    a_j = tl.load(a_ptr + j_offsets, mask=mask_j, other=0.0)
    b_i = tl.load(b_ptr + i_offsets, mask=mask_i, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result
    tl.store(a_ptr + i_offsets, result, mask=mask_i)

def s121_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](a, b, n_elements, BLOCK_SIZE)
    
    return a