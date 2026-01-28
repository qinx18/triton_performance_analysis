import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load d[i] and e[i]
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute b[i] = d[i] * e[i]
    b_vals = d_vals * e_vals
    
    # Store b[i]
    tl.store(b_ptr + indices, b_vals, mask=mask)
    
    # Load c[i] for s152s computation
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    # Load a[i] for s152s computation
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute a[i] += b[i] * c[i] (s152s function)
    a_vals = a_vals + b_vals * c_vals
    
    # Store updated a[i]
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a