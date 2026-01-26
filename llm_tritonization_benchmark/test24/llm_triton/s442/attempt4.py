import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < N
    
    # Load indices
    indx_offsets = block_start + offsets
    indices = tl.load(indx_ptr + indx_offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + indx_offsets, mask=mask, other=0.0)
    
    # Load arrays b, c, d, e
    b_vals = tl.load(b_ptr + indx_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indx_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indx_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indx_offsets, mask=mask, other=0.0)
    
    # Compute squares
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Select which value to add based on index, but make sure all valid indices get an addition
    valid_indices = (indices >= 1) & (indices <= 4)
    add_val = tl.where(indices == 1, b_squared,
              tl.where(indices == 2, c_squared,
              tl.where(indices == 3, d_squared,
              tl.where(indices == 4, e_squared, 0.0))))
    
    # Only add for valid indices, otherwise keep original value
    final_result = tl.where(valid_indices, a_vals + add_val, a_vals)
    
    # Store result
    tl.store(a_ptr + indx_offsets, final_result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a