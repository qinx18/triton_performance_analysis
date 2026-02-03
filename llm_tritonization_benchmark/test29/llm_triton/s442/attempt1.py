import triton
import triton.language as tl
import torch

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
    
    # Load index values
    indx_vals = tl.load(indx_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load input arrays
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute updates based on index values
    case1_mask = indx_vals == 1
    case2_mask = indx_vals == 2
    case3_mask = indx_vals == 3
    case4_mask = indx_vals == 4
    
    update_vals = tl.where(case1_mask, b_vals * b_vals,
                  tl.where(case2_mask, c_vals * c_vals,
                  tl.where(case3_mask, d_vals * d_vals,
                  tl.where(case4_mask, e_vals * e_vals, 0.0))))
    
    # Update a array
    result_vals = a_vals + update_vals
    tl.store(a_ptr + block_start + offsets, result_vals, mask=mask)

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