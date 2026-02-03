import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # k = 2*i + 1 (from the pattern j=-1, k=j+1, j=k+1)
    k_offsets = 2 * i_offsets + 1
    k_mask = mask
    
    # Load values
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + k_offsets, mask=k_mask, other=0.0)
    c_vals = tl.load(c_ptr + k_offsets, mask=k_mask, other=0.0)
    
    # a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)
    
    # b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    tl.store(b_ptr + k_offsets, b_new_vals, mask=k_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )