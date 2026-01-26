import triton
import triton.language as tl
import torch

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Conditional logic - need to check d[i] exactly as in C code
    # if (d[i] < 0.) -> a[i] += b[i] * c[i]
    # else if (d[i] == 0.) -> a[i] += b[i] * b[i]  
    # else -> a[i] += c[i] * c[i]
    
    neg_mask = d_vals < 0.0
    zero_mask = (d_vals == 0.0) & (~neg_mask)
    pos_mask = (~neg_mask) & (~zero_mask)
    
    # Compute updates
    update = tl.where(neg_mask, b_vals * c_vals,
                     tl.where(zero_mask, b_vals * b_vals, c_vals * c_vals))
    
    # Apply update to a
    result = a_vals + update
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s441_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s441_kernel[grid](
        a, b, c, d,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )