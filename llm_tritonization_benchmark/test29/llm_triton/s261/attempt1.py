import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements - 1
    
    # Load data for indices 1 to n_elements-1
    actual_idx = idx + 1
    load_mask = actual_idx < n_elements
    
    a_vals = tl.load(a_ptr + actual_idx, mask=load_mask)
    b_vals = tl.load(b_ptr + actual_idx, mask=load_mask)
    c_vals = tl.load(c_ptr + actual_idx - 1, mask=load_mask)  # c[i-1]
    c_current = tl.load(c_ptr + actual_idx, mask=load_mask)   # c[i]
    d_vals = tl.load(d_ptr + actual_idx, mask=load_mask)
    
    # t = a[i] + b[i]
    # a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    new_a = t1 + c_vals
    
    # t = c[i] * d[i]
    # c[i] = t
    t2 = c_current * d_vals
    new_c = t2
    
    # Store results
    tl.store(a_ptr + actual_idx, new_a, mask=load_mask)
    tl.store(c_ptr + actual_idx, new_c, mask=load_mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    if n_elements <= 1:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )