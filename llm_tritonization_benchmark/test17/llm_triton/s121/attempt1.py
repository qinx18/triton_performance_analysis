import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load b[i]
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Load a[j] where j = i + 1
    j_idx = idx + 1
    j_mask = j_idx < (n_elements + 1)  # j can go up to n_elements
    a_j_vals = tl.load(a_ptr + j_idx, mask=j_mask)
    
    # Compute a[i] = a[j] + b[i]
    result = a_j_vals + b_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s121_triton(a, b):
    n_elements = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s121_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a