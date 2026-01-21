import triton
import triton.language as tl
import torch

@triton.jit
def s127_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Each thread processes one i value
    i_offsets = block_start + offsets
    mask = i_offsets < n // 2
    
    # Load b[i], c[i], d[i], e[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    
    # Compute values for a[2*i] and a[2*i+1]
    # j = 2*i for first assignment, j = 2*i+1 for second assignment
    a_val1 = b_vals + c_vals * d_vals  # a[2*i]
    a_val2 = b_vals + d_vals * e_vals  # a[2*i+1]
    
    # Store to a[2*i] and a[2*i+1]
    j_offsets1 = 2 * i_offsets
    j_offsets2 = 2 * i_offsets + 1
    
    mask1 = j_offsets1 < n
    mask2 = j_offsets2 < n
    
    tl.store(a_ptr + j_offsets1, a_val1, mask=mask & mask1)
    tl.store(a_ptr + j_offsets2, a_val2, mask=mask & mask2)

def s127_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    s127_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE=BLOCK_SIZE)