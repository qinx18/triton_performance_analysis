import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    i_offsets = block_start + offsets
    j_offsets = i_offsets + 1
    
    # Load a[j] and b[i]
    a_j = tl.load(a_ptr + j_offsets, mask=mask)
    b_i = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j + b_i
    
    # Store result to a[i]
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )