import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute values for both j positions
    # For j = 2*i (even positions): a[j] = b[i] + c[i] * d[i]
    val_even = b_vals + c_vals * d_vals
    # For j = 2*i+1 (odd positions): a[j] = b[i] + d[i] * e[i]
    val_odd = b_vals + d_vals * e_vals
    
    # Store to even positions (j = 2*i)
    even_offsets = 2 * offsets
    even_mask = mask & (even_offsets < n_elements * 2)
    tl.store(a_ptr + even_offsets, val_even, mask=even_mask)
    
    # Store to odd positions (j = 2*i+1)
    odd_offsets = 2 * offsets + 1
    odd_mask = mask & (odd_offsets < n_elements * 2)
    tl.store(a_ptr + odd_offsets, val_odd, mask=odd_mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2 iterations
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )