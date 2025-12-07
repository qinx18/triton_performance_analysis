import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    a_copy_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
    
    # S0: a[i] = b[i] + c[i] * d[i]
    result_0 = b_vals + c_vals * d_vals
    tl.store(a_ptr + current_offsets, result_0, mask=mask)
    
    # S1: b[i] = c[i] + b[i]
    result_1 = c_vals + b_vals
    tl.store(b_ptr + current_offsets, result_1, mask=mask)
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (using updated b values)
    result_2 = result_1 + a_copy_vals * d_vals
    tl.store(a_ptr + current_offsets + 1, result_2, mask=mask)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )