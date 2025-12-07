import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load a[j] where j = i + 1, so a[i+1]
    a_j_indices = indices + 1
    a_j_mask = a_j_indices < (n_elements + 1)  # Account for the +1 offset
    a_j = tl.load(a_ptr + a_j_indices, mask=a_j_mask, other=0.0)
    
    # Load b[i]
    b_i = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result to a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )