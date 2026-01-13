import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Calculate j indices for the two stores
    # j starts at -1, then increments twice per i iteration
    # For i-th iteration: j = 2*i and j = 2*i + 1
    j_offsets_1 = 2 * i_offsets
    j_offsets_2 = 2 * i_offsets + 1
    
    # Calculate values to store
    val1 = b_vals + c_vals * d_vals
    val2 = b_vals + d_vals * e_vals
    
    # Store to a array
    # Need to check bounds for both j indices
    mask_1 = (i_offsets < n_elements) & (j_offsets_1 < 2 * n_elements)
    mask_2 = (i_offsets < n_elements) & (j_offsets_2 < 2 * n_elements)
    
    tl.store(a_ptr + j_offsets_1, val1, mask=mask_1)
    tl.store(a_ptr + j_offsets_2, val2, mask=mask_2)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )