import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute first values: a[2*i] = b[i] + c[i] * d[i]
    a_vals_0 = b_vals + c_vals * d_vals
    # Compute second values: a[2*i+1] = b[i] + d[i] * e[i]
    a_vals_1 = b_vals + d_vals * e_vals
    
    # Store to a array with stride pattern
    a_offsets_0 = 2 * offsets
    a_offsets_1 = 2 * offsets + 1
    a_mask_0 = mask & (a_offsets_0 < 2 * n_elements)
    a_mask_1 = mask & (a_offsets_1 < 2 * n_elements)
    
    tl.store(a_ptr + a_offsets_0, a_vals_0, mask=a_mask_0)
    tl.store(a_ptr + a_offsets_1, a_vals_1, mask=a_mask_1)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )