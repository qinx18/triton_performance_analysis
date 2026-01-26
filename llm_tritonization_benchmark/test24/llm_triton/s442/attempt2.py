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
    
    # Select which array to add based on index
    mask_case1 = (indices == 1) & mask
    mask_case2 = (indices == 2) & mask
    mask_case3 = (indices == 3) & mask
    mask_case4 = (indices == 4) & mask
    
    # Compute the final result based on the switch logic
    final_result = tl.where(mask_case1, a_vals + b_squared,
                   tl.where(mask_case2, a_vals + c_squared,
                   tl.where(mask_case3, a_vals + d_squared,
                   tl.where(mask_case4, a_vals + e_squared, a_vals))))
    
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