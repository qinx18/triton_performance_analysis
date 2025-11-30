import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute conditionally
    # if (d[i] < 0.) a[i] += b[i] * c[i];
    # else if (d[i] == 0.) a[i] += b[i] * b[i];
    # else a[i] += c[i] * c[i];
    
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = ~(neg_mask | zero_mask)
    
    result = a_vals
    result = tl.where(neg_mask, result + b_vals * c_vals, result)
    result = tl.where(zero_mask, result + b_vals * b_vals, result)
    result = tl.where(pos_mask, result + c_vals * c_vals, result)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )