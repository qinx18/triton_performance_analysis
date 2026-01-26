import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < N
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    indx_vals = tl.load(indx_ptr + idx, mask=mask)
    
    # Compute squared values
    b_sq = b_vals * b_vals
    c_sq = c_vals * c_vals
    d_sq = d_vals * d_vals
    e_sq = e_vals * e_vals
    
    # Create condition masks
    is_case1 = indx_vals == 1
    is_case2 = indx_vals == 2
    is_case3 = indx_vals == 3
    is_case4 = indx_vals == 4
    
    # Compute updates based on conditions
    update = tl.where(is_case1, b_sq,
             tl.where(is_case2, c_sq,
             tl.where(is_case3, d_sq,
             tl.where(is_case4, e_sq, 0.0))))
    
    # Update a array
    result = a_vals + update
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a