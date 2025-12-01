import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements - 1  # Ensure we don't exceed LEN_1D-1
    
    # Load values for a[i] = b[i - 1] + c[i] * d[i]
    b_prev_offsets = offsets - 1
    c_offsets = offsets
    d_offsets = offsets
    
    b_prev_vals = tl.load(b_copy_ptr + b_prev_offsets, mask=mask)
    c_vals = tl.load(c_ptr + c_offsets, mask=mask)
    d_vals = tl.load(d_ptr + d_offsets, mask=mask)
    
    # Load values for b[i] = b[i + 1] - e[i] * d[i]
    b_next_offsets = offsets + 1
    e_offsets = offsets
    
    b_next_vals = tl.load(b_copy_ptr + b_next_offsets, mask=mask)
    e_vals = tl.load(e_ptr + e_offsets, mask=mask)
    
    # Compute results
    a_result = b_prev_vals + c_vals * d_vals
    b_result = b_next_vals - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for b array to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 2, BLOCK_SIZE),)  # -2 because we process indices 1 to LEN_1D-2
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )