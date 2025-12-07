import triton
import triton.language as tl
import torch

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
    
    # Compute the two expressions
    val1 = b_vals + c_vals * d_vals
    val2 = b_vals + d_vals * e_vals
    
    # Store to consecutive pairs in a
    j_offsets_1 = 2 * i_offsets
    j_offsets_2 = 2 * i_offsets + 1
    
    # Masks for output arrays (j ranges from 0 to n_elements*2-1)
    mask_1 = j_offsets_1 < (n_elements * 2)
    mask_2 = j_offsets_2 < (n_elements * 2)
    
    tl.store(a_ptr + j_offsets_1, val1, mask=mask & mask_1)
    tl.store(a_ptr + j_offsets_2, val2, mask=mask & mask_2)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a