import torch
import triton
import triton.language as tl

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = (idx >= 1) & (idx < n_elements - 1)
    
    # Load values
    b_prev = tl.load(b_copy_ptr + (idx - 1), mask=mask)
    b_next = tl.load(b_copy_ptr + (idx + 1), mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute results
    a_result = b_prev + c_vals * d_vals
    b_result = b_next - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_result, mask=mask)
    tl.store(b_ptr + idx, b_result, mask=mask)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create read-only copy for WAR dependency handling
    b_copy = b.clone()
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )