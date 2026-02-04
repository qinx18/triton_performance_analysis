import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    
    # Main range: i from 1 to N-2
    valid_mask = (offsets >= 1) & (offsets < N - 1)
    
    # Load data for all valid positions
    b_prev = tl.load(b_copy_ptr + offsets - 1, mask=valid_mask)
    b_next = tl.load(b_copy_ptr + offsets + 1, mask=valid_mask)
    c_vals = tl.load(c_ptr + offsets, mask=valid_mask)
    d_vals = tl.load(d_ptr + offsets, mask=valid_mask)
    e_vals = tl.load(e_ptr + offsets, mask=valid_mask)
    
    # Compute both statements
    a_vals = b_prev + c_vals * d_vals
    b_vals = b_next - e_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=valid_mask)
    tl.store(b_ptr + offsets, b_vals, mask=valid_mask)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy of b to avoid WAR race condition
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        N, BLOCK_SIZE
    )