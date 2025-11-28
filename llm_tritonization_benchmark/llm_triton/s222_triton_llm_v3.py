import triton
import triton.language as tl
import torch

@triton.jit
def s222_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    e_ptr,
    e_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_prev_vals = tl.load(e_copy_ptr + offsets - 1, mask=mask)
    
    # Compute
    bc_product = b_vals * c_vals
    a_temp = a_vals + bc_product
    e_new = e_prev_vals * e_prev_vals
    a_final = a_temp - bc_product
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(e_ptr + offsets, e_new, mask=mask)

def s222_triton(a, b, c, e):
    n_elements = a.shape[0]
    
    # Create read-only copy of e to handle WAR dependency
    e_copy = e.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s222_kernel[grid](
        a,
        b, 
        c,
        e,
        e_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )