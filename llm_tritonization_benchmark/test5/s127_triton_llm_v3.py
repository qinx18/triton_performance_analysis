import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    mask = i_offsets < n
    
    # Load values for iteration i
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    # j = i*2 (first increment)
    j_offsets_1 = i_offsets * 2
    # j = i*2 + 1 (second increment)
    j_offsets_2 = i_offsets * 2 + 1
    
    # Compute values
    val1 = b_vals + c_vals * d_vals
    val2 = b_vals + d_vals * e_vals
    
    # Store results
    mask_j1 = j_offsets_1 < (n * 2)
    mask_j2 = j_offsets_2 < (n * 2)
    
    tl.store(a_ptr + j_offsets_1, val1, mask=mask & mask_j1)
    tl.store(a_ptr + j_offsets_2, val2, mask=mask & mask_j2)

def s127_triton(a, b, c, d, e):
    n = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )