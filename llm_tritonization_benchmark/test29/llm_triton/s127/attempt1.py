import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute base offset
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    
    # Create offset vectors
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = base_offset + offsets
    
    # Mask for valid elements
    mask = i_offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute output values
    # j starts at -1, then increments to 0, 1, 2, 3, ...
    # For i-th iteration: j=2*i and j=2*i+1
    val1 = b_vals + c_vals * d_vals  # a[2*i] = b[i] + c[i] * d[i]
    val2 = b_vals + d_vals * e_vals  # a[2*i+1] = b[i] + d[i] * e[i]
    
    # Store results at corresponding j positions
    j_offsets1 = 2 * i_offsets      # j = 2*i
    j_offsets2 = 2 * i_offsets + 1  # j = 2*i + 1
    
    # Store first set of values
    tl.store(a_ptr + j_offsets1, val1, mask=mask)
    # Store second set of values
    tl.store(a_ptr + j_offsets2, val2, mask=mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # Loop runs for LEN_1D/2 iterations
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )