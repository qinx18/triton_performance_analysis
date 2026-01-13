import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices for this block
    i_offsets = block_start + offsets
    j_offsets = i_offsets + 1
    
    # Create masks for valid elements
    i_mask = i_offsets < n_elements
    j_mask = j_offsets < (n_elements + 1)  # j can go one element beyond n_elements
    
    # Load data
    a_j = tl.load(a_ptr + j_offsets, mask=j_mask, other=0.0)
    b_i = tl.load(b_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2 (n_elements-1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )