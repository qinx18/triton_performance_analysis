import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # We need to process indices starting from 1
    current_indices = offsets + 1
    mask = current_indices < n_elements
    
    # Load values at current indices
    a_vals = tl.load(a_ptr + current_indices, mask=mask)
    b_vals = tl.load(b_ptr + current_indices, mask=mask)
    c_vals = tl.load(c_ptr + current_indices, mask=mask)
    
    # Load c[i-1] values
    prev_indices = current_indices - 1
    c_prev_vals = tl.load(c_ptr + prev_indices, mask=mask)
    
    # Load d values
    d_vals = tl.load(d_ptr + current_indices, mask=mask)
    
    # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    a_new = t1 + c_prev_vals
    
    # Compute: t = c[i] * d[i]; c[i] = t
    t2 = c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + current_indices, a_new, mask=mask)
    tl.store(c_ptr + current_indices, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    # We process from index 1 to n_elements-1
    num_elements_to_process = n_elements - 1
    grid = (triton.cdiv(num_elements_to_process, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )