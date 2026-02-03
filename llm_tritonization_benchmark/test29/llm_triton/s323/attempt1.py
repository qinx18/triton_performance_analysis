import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process blocks sequentially due to dependencies
    for start_idx in range(1, N-1, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < N-1
        
        # Load data for b[i] computation
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask) 
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute and store b[i]
        b_vals = a_vals + c_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # Load data for a[i+1] computation
        next_offsets = current_offsets + 1
        next_mask = next_offsets < N
        
        b_prev = tl.load(b_ptr + current_offsets, mask=mask)
        c_next = tl.load(c_ptr + next_offsets, mask=next_mask)
        d_next = tl.load(d_ptr + next_offsets, mask=next_mask)
        
        # Compute and store a[i+1]
        a_next = b_prev + c_next * d_next
        tl.store(a_ptr + next_offsets, a_next, mask=next_mask)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    if N <= 1:
        return
    
    # Handle prologue: a[1] = b[0] + c[1] * d[1]
    if N > 1:
        a[1] = b[0] + c[1] * d[1]
    
    # Handle epilogue: b[N-1] = a[N-1] + c[N-1] * e[N-1] 
    if N > 1:
        b[N-1] = a[N-1] + c[N-1] * e[N-1]
    
    # Main computation with single thread for dependency handling
    BLOCK_SIZE = min(1024, N)
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )