import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Start from index 1, so adjust offsets
    offsets = offsets + 1
    mask = offsets < n_elements
    
    # Load current elements
    a_val = tl.load(a_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr + offsets, mask=mask)
    c_val = tl.load(c_ptr + offsets, mask=mask)
    d_val = tl.load(d_ptr + offsets, mask=mask)
    
    # Load c[i-1] values
    c_prev_offsets = offsets - 1
    c_prev_mask = (offsets >= 1) & mask
    c_prev_val = tl.load(c_ptr + c_prev_offsets, mask=c_prev_mask)
    
    # Compute: t = a[i] + b[i]; a[i] = t + c[i-1]
    t1 = a_val + b_val
    new_a = t1 + c_prev_val
    
    # Compute: t = c[i] * d[i]; c[i] = t
    t2 = c_val * d_val
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(c_ptr + offsets, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # We start from index 1, so we process n_elements - 1 elements
    elements_to_process = n_elements - 1
    
    if elements_to_process <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(elements_to_process, meta['BLOCK_SIZE']),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )