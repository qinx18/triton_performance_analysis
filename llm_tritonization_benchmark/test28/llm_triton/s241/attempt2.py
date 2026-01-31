import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load data for current positions
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Load a[i+1] from copy for the second computation
    offset_plus_1 = current_offsets + 1
    mask_plus_1 = offset_plus_1 < (n_elements + 1)  # Allow reading a[i+1] where i < LEN_1D-1
    a_plus_1_vals = tl.load(a_copy_ptr + offset_plus_1, mask=mask_plus_1, other=0.0)
    
    # First computation: a[i] = b[i] * c[i] * d[i]
    new_a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
    
    # Second computation: b[i] = a[i] * a[i+1] * d[i]
    new_b_vals = new_a_vals * a_plus_1_vals * d_vals
    tl.store(b_ptr + current_offsets, new_b_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop runs for LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )