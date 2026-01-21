import torch
import triton
import triton.language as tl

@triton.jit
def s1111_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load input values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Compute: c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i]
    result = (c_vals * b_vals + 
              d_vals * b_vals + 
              c_vals * c_vals + 
              d_vals * b_vals + 
              d_vals * c_vals)
    
    # Store to a[2*i] positions
    output_indices = 2 * indices
    output_mask = mask & (output_indices < (2 * n))
    
    tl.store(a_ptr + output_indices, result, mask=output_mask)

def s1111_triton(a, b, c, d):
    # Derive dimensions from input tensors
    full_len = a.shape[0]
    n = full_len // 2  # Loop runs for LEN_1D/2 iterations
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1111_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )