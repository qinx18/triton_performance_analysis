import triton
import triton.language as tl
import torch

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    
    # Apply condition: if b[i] > 0.0
    condition_mask = b_vals > 0.0
    combined_mask = mask & condition_mask
    
    # Compute: a[i] += b[i] * c[i] where condition is true
    result = tl.where(condition_mask, a_vals + b_vals * c_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s271_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s271_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )