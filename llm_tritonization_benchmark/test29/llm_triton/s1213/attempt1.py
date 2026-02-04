import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel_loop1(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n-1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets >= 1) & (current_offsets < n-1)
        
        # Load b[i-1], c[i]
        b_vals = tl.load(b_ptr + current_offsets - 1, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute a[i] = b[i-1] + c[i]
        a_vals = b_vals + c_vals
        
        # Store a[i]
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

@triton.jit
def s1213_kernel_loop2(a_copy_ptr, b_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n-1, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = (current_offsets >= 1) & (current_offsets < n-1)
        
        # Load a[i+1], d[i]
        a_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Compute b[i] = a[i+1] * d[i]
        b_vals = a_vals * d_vals
        
        # Store b[i]
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    
    # Launch first kernel for statement S0: a[i] = b[i-1] + c[i]
    s1213_kernel_loop1[(1,)](
        a, a_copy, b, c, n, BLOCK_SIZE
    )
    
    # Launch second kernel for statement S1: b[i] = a[i+1] * d[i]
    s1213_kernel_loop2[(1,)](
        a_copy, b, d, n, BLOCK_SIZE
    )