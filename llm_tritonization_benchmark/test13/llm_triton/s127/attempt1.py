import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid elements
    mask = i_offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute j indices: j starts at -1, increments twice per i
    # First increment: j = 2*i
    # Second increment: j = 2*i + 1
    j_offsets_first = 2 * i_offsets
    j_offsets_second = 2 * i_offsets + 1
    
    # Compute values
    # a[j] = b[i] + c[i] * d[i]
    val_first = b_vals + c_vals * d_vals
    # a[j] = b[i] + d[i] * e[i]
    val_second = b_vals + d_vals * e_vals
    
    # Store results
    tl.store(a_ptr + j_offsets_first, val_first, mask=mask)
    tl.store(a_ptr + j_offsets_second, val_second, mask=mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a