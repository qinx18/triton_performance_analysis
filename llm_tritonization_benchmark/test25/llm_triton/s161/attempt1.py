import triton
import triton.language as tl
import torch

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        else_mask = b_vals < 0.0
        combined_mask = mask & else_mask
        
        a_vals = tl.load(a_ptr + current_offsets, mask=combined_mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=combined_mask)
        
        result = a_vals + d_vals * d_vals
        
        write_offsets = current_offsets + 1
        write_mask = combined_mask & (write_offsets < (n + 1))
        tl.store(c_ptr + write_offsets, result, mask=write_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        if_mask = b_vals >= 0.0
        combined_mask = mask & if_mask
        
        c_vals = tl.load(c_ptr + current_offsets, mask=combined_mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=combined_mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=combined_mask)
        
        result = c_vals + d_vals * e_vals
        
        tl.store(a_ptr + current_offsets, result, mask=combined_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    # Step 1: Process else branch first (c[i+1] = a[i] + d[i] * d[i])
    s161_kernel_else[(triton.cdiv(n, BLOCK_SIZE),)](
        a, b, c, d, n, BLOCK_SIZE
    )
    
    # Step 2: Process if branch second (a[i] = c[i] + d[i] * e[i])
    s161_kernel_if[(triton.cdiv(n, BLOCK_SIZE),)](
        a, b, c, d, e, n, BLOCK_SIZE
    )