import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    mask_i = i_offsets < n_elements
    
    # Load input values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask_i)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask_i)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask_i)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask_i)
    
    # Compute output values
    # j starts at 0 for first element (j=-1 then j++)
    # For i-th iteration: j = 2*i and j = 2*i+1
    j_even = 2 * i_offsets  # j after first increment
    j_odd = 2 * i_offsets + 1  # j after second increment
    
    # First computation: a[j] = b[i] + c[i] * d[i]
    val1 = b_vals + c_vals * d_vals
    mask_j_even = j_even < (n_elements * 2)
    tl.store(a_ptr + j_even, val1, mask=mask_i & mask_j_even)
    
    # Second computation: a[j] = b[i] + d[i] * e[i]
    val2 = b_vals + d_vals * e_vals
    mask_j_odd = j_odd < (n_elements * 2)
    tl.store(a_ptr + j_odd, val2, mask=mask_i & mask_j_odd)

def s127_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2 iterations
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s127_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )