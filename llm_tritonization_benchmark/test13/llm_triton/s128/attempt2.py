import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Compute j and k values for all elements in block
    # j starts at -1, increments by 2 each iteration: j = 2*i - 1
    # k = j + 1 = 2*i
    k_offsets = 2 * i_offsets
    
    # Load values with masks
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    b_vals = tl.load(b_ptr + k_offsets, mask=mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask)
    
    # First computation: a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)
    
    # Second computation: b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c, d