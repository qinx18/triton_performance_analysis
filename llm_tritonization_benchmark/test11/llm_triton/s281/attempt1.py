import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    i_offsets = block_start + offsets
    reverse_offsets = n_elements - 1 - i_offsets
    
    # Load data
    a_reverse = tl.load(a_ptr + reverse_offsets, mask=mask)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a_reverse + b_vals * c_vals
    
    # Store results
    tl.store(a_ptr + i_offsets, x - 1.0, mask=mask)
    tl.store(b_ptr + i_offsets, x, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s281_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)