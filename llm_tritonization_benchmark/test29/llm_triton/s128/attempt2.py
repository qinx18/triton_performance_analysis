import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes a chunk of the loop iterations
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # For each element i, k = 2*i (since j starts at -1, increments by 2)
    k_offsets = 2 * offsets
    
    # Load values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + k_offsets, mask=mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask)
    
    # a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    N = a.shape[0]
    n_elements = N // 2  # Loop runs for LEN_1D/2 iterations
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )