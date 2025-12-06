import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in this block
    i_offsets = block_start + offsets
    j_offsets = i_offsets + 1
    
    # Mask for valid elements (i < n_elements-1, so j < n_elements)
    mask_i = i_offsets < n_elements - 1
    mask_j = j_offsets < n_elements
    
    # Load a[j] and b[i]
    a_j_vals = tl.load(a_ptr + j_offsets, mask=mask_j, other=0.0)
    b_i_vals = tl.load(b_ptr + i_offsets, mask=mask_i, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j_vals + b_i_vals
    
    # Store result to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask_i)

def s121_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a