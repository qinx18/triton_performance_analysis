import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(
    a_ptr,
    b_ptr,
    b_copy_ptr,
    c_ptr,
    d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_prev_vals = tl.load(b_copy_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute a[i] += c[i] * d[i]
    new_a_vals = a_vals + c_vals * d_vals
    
    # Compute b[i] = b[i-1] + a[i] + d[i]
    new_b_vals = b_prev_vals + new_a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)

def s221_triton(a, b, c, d):
    n_elements = a.numel()
    
    # Create read-only copy for WAR dependency
    b_copy = b.clone()
    
    # Process elements starting from index 1
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s221_kernel[grid](
        a,
        b,
        b_copy,
        c,
        d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )