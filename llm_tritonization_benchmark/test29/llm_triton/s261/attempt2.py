import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets + 1
    
    mask = indices < n_elements
    
    # Load values for current indices
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    
    # Load c[i-1] values
    prev_indices = indices - 1
    prev_mask = (prev_indices >= 0) & mask
    c_prev_vals = tl.load(c_ptr + prev_indices, mask=prev_mask, other=0.0)
    
    # Compute: t = a[i] + b[i], a[i] = t + c[i-1]
    t1 = a_vals + b_vals
    new_a = t1 + c_prev_vals
    
    # Compute: t = c[i] * d[i], c[i] = t
    t2 = c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, new_a, mask=mask)
    tl.store(c_ptr + indices, t2, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )