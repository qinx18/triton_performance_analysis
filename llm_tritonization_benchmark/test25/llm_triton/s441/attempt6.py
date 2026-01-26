import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Conditional logic: if d[i] < 0, elif d[i] == 0, else
    neg_mask = d_vals < 0.0
    zero_mask = d_vals == 0.0
    pos_mask = (d_vals > 0.0)
    
    # Apply masks to ensure only one condition per element
    neg_mask = neg_mask & mask
    zero_mask = zero_mask & (~neg_mask) & mask
    pos_mask = pos_mask & (~neg_mask) & (~zero_mask) & mask
    
    # Calculate updates
    update1 = b_vals * c_vals  # for d[i] < 0
    update2 = b_vals * b_vals  # for d[i] == 0
    update3 = c_vals * c_vals  # for d[i] > 0
    
    # Apply conditional updates
    result = a_vals
    result = tl.where(neg_mask, result + update1, result)
    result = tl.where(zero_mask, result + update2, result)
    result = tl.where(pos_mask, result + update3, result)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)