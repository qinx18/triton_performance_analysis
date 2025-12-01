import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, a_out_ptr, b_ptr, c_ptr, k, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Load a[i + k]
    a_offset_indices = indices + k
    a_offset_mask = mask & (a_offset_indices < (n_elements + k))
    a_offset_vals = tl.load(a_ptr + a_offset_indices, mask=a_offset_mask, other=0.0)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_offset_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_out_ptr + indices, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n_elements = len(a) - 1
    a_out = torch.empty_like(a)
    a_out.copy_(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_out, b, c, k, n_elements, BLOCK_SIZE
    )
    
    a.copy_(a_out)