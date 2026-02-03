import torch
import triton
import triton.language as tl

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < N
    
    # Load arrays
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Compute s values using scalar expansion
    # For i=0: s = 0.0
    # For i>0: s = b[i-1] + c[i-1]
    s_vals = tl.zeros_like(b_vals)
    
    # For i > 0, load previous b and c values
    prev_mask = (current_offsets > 0) & mask
    prev_offsets = current_offsets - 1
    prev_b_vals = tl.load(b_ptr + prev_offsets, mask=prev_mask)
    prev_c_vals = tl.load(c_ptr + prev_offsets, mask=prev_mask)
    
    # Set s values: 0.0 for i=0, b[i-1]+c[i-1] for i>0
    s_vals = tl.where(current_offsets > 0, prev_b_vals + prev_c_vals, 0.0)
    
    # Compute a[i] = s * e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i] + d[i]
    b_new_vals = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2251_kernel[grid](
        a, b, c, d, e, N, BLOCK_SIZE
    )