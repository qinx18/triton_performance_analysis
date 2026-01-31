import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Compute k = 2*i for each element
    k_offsets = 2 * offsets
    
    # Load values with masking
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + k_offsets, mask=mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s128_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )