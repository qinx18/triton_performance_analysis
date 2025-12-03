import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_val = tl.load(d_ptr + idx, mask=mask, other=0.0)
    
    # Check if any d[i] < 0 - if so, we would exit in original code
    # In GPU context, we'll skip computation for those elements
    valid_mask = mask & (d_val >= 0.0)
    
    # Load other arrays only for valid elements
    a_val = tl.load(a_ptr + idx, mask=valid_mask, other=0.0)
    b_val = tl.load(b_ptr + idx, mask=valid_mask, other=0.0)
    c_val = tl.load(c_ptr + idx, mask=valid_mask, other=0.0)
    
    # Compute a[i] += b[i] * c[i]
    result = a_val + b_val * c_val
    
    # Store result
    tl.store(a_ptr + idx, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )