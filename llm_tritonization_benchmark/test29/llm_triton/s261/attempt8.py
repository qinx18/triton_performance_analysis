import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Parallel computation for indices 1 to n-2
    # Each thread computes c[i] and a[i+1] using c[i]
    i_offsets = offsets + 1
    mask = i_offsets < (n - 1)
    
    # Load data for index i
    a_i = tl.load(a_ptr + i_offsets, mask=mask)
    b_i = tl.load(b_ptr + i_offsets, mask=mask)
    c_i = tl.load(c_ptr + i_offsets, mask=mask)
    d_i = tl.load(d_ptr + i_offsets, mask=mask)
    
    # Compute c[i] = c[i] * d[i]
    new_c_i = c_i * d_i
    
    # Store c[i]
    tl.store(c_ptr + i_offsets, new_c_i, mask=mask)
    
    # Compute a[i+1] = (a[i+1] + b[i+1]) + c[i]
    # Need to load a[i+1] and b[i+1]
    next_i_offsets = i_offsets + 1
    next_mask = next_i_offsets < n
    
    a_next = tl.load(a_ptr + next_i_offsets, mask=next_mask)
    b_next = tl.load(b_ptr + next_i_offsets, mask=next_mask)
    
    new_a_next = (a_next + b_next) + new_c_i
    
    # Store a[i+1]
    tl.store(a_ptr + next_i_offsets, new_a_next, mask=next_mask)

@triton.jit
def s261_prologue_kernel(a_ptr, b_ptr, c_ptr):
    # Handle prologue: a[1] = (a[1] + b[1]) + c[0]
    a_1 = tl.load(a_ptr + 1)
    b_1 = tl.load(b_ptr + 1)
    c_0 = tl.load(c_ptr + 0)
    
    new_a_1 = (a_1 + b_1) + c_0
    tl.store(a_ptr + 1, new_a_1)

@triton.jit
def s261_epilogue_kernel(c_ptr, d_ptr, n):
    # Handle epilogue: c[n-1] = c[n-1] * d[n-1]
    idx = n - 1
    c_last = tl.load(c_ptr + idx)
    d_last = tl.load(d_ptr + idx)
    
    new_c_last = c_last * d_last
    tl.store(c_ptr + idx, new_c_last)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    if n <= 1:
        return
    
    # Prologue: handle a[1] using original c[0]
    grid = (1,)
    s261_prologue_kernel[grid](a, b, c)
    
    # Main parallel loop for i from 1 to n-2
    if n > 2:
        num_blocks = triton.cdiv(n - 2, BLOCK_SIZE)
        grid = (num_blocks,)
        s261_kernel[grid](a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Epilogue: handle c[n-1]
    if n > 1:
        grid = (1,)
        s261_epilogue_kernel[grid](c, d, n)