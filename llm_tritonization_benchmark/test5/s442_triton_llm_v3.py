import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    indices = tl.load(indx_ptr + block_start + offsets, mask=mask)
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Compute updates based on index values
    update_vals = tl.where(indices == 1, b_vals * b_vals,
                  tl.where(indices == 2, c_vals * c_vals,
                  tl.where(indices == 3, d_vals * d_vals,
                  tl.where(indices == 4, e_vals * e_vals, 0.0))))
    
    # Update a array
    result = a_vals + update_vals
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a