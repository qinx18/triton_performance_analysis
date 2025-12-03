import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Conditional computation using tl.where
    # if d[i] < 0.0: a[i] += b[i] * c[i]
    # else if d[i] == 0.0: a[i] += b[i] * b[i]  
    # else: a[i] += c[i] * c[i]
    
    cond1 = d_vals < 0.0
    cond2 = d_vals == 0.0
    
    term1 = b_vals * c_vals  # for d < 0
    term2 = b_vals * b_vals  # for d == 0
    term3 = c_vals * c_vals  # for d > 0
    
    # Use nested tl.where to implement if-else if-else logic
    add_term = tl.where(cond1, term1, tl.where(cond2, term2, term3))
    
    result = a_vals + add_term
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s441_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s441_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)