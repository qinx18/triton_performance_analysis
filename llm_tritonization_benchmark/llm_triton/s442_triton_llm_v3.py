import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices
    indices = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load array values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute based on switch/goto logic
    # case 1: a[i] += b[i] * b[i]
    # case 2: a[i] += c[i] * c[i]
    # case 3: a[i] += d[i] * d[i]
    # case 4: a[i] += e[i] * e[i]
    
    updates = tl.where(indices == 1, b_vals * b_vals,
              tl.where(indices == 2, c_vals * c_vals,
              tl.where(indices == 3, d_vals * d_vals,
              tl.where(indices == 4, e_vals * e_vals, 0.0))))
    
    result = a_vals + updates
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a