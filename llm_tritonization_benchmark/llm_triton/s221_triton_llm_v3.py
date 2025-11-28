import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements) & (offsets >= 1)
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_prev_vals = tl.load(b_copy_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute
    a_new = a_vals + c_vals * d_vals
    b_new = b_prev_vals + a_new + d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s221_kernel[grid](
        a, b, b_copy, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )