import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, b_ptr, c_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b and c
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Load a[i + k] with bounds checking
    a_offset_offsets = offsets + k
    a_offset_mask = mask & (a_offset_offsets < (n_elements + k))
    a_offset = tl.load(a_ptr + a_offset_offsets, mask=a_offset_mask)
    
    # Compute result
    result = a_offset + b * c
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n_elements = len(a) - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](a, b, c, k, n_elements, BLOCK_SIZE)