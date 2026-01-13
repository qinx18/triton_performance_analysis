import torch
import triton
import triton.language as tl

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    i_mask = i_offsets < n_elements
    
    # j starts at -1 and increments by 2 each iteration
    # For i-th iteration: j = 2*i and j = 2*i+1
    j_offsets_0 = 2 * i_offsets
    j_offsets_1 = 2 * i_offsets + 1
    
    # Load arrays b, c, d, e at positions i
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=i_mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=i_mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=i_mask)
    
    # First computation: a[j] = b[i] + c[i] * d[i]
    result_0 = b_vals + c_vals * d_vals
    tl.store(a_ptr + j_offsets_0, result_0, mask=i_mask)
    
    # Second computation: a[j+1] = b[i] + d[i] * e[i]  
    result_1 = b_vals + d_vals * e_vals
    tl.store(a_ptr + j_offsets_1, result_1, mask=i_mask)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )