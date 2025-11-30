import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Load a[i + k]
    read_offsets = offsets + k
    read_mask = mask & (read_offsets < (n_elements + k))
    a_read_vals = tl.load(a_ptr + read_offsets, mask=read_mask)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_read_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](a, b, c, k, n_elements, BLOCK_SIZE)