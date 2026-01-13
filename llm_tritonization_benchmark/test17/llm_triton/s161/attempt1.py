import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values for current block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Check condition: b[i] < 0
        else_mask = b_vals < 0.0
        combined_mask = mask & else_mask
        
        # c[i+1] = a[i] + d[i] * d[i]
        if tl.sum(combined_mask.to(tl.int32)) > 0:
            result = a_vals + d_vals * d_vals
            tl.store(c_ptr + current_offsets + 1, result, mask=combined_mask)

@triton.jit
def s161_kernel_if(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values for current block
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Check condition: b[i] >= 0
        if_mask = b_vals >= 0.0
        combined_mask = mask & if_mask
        
        # a[i] = c[i] + d[i] * e[i]
        if tl.sum(combined_mask.to(tl.int32)) > 0:
            result = c_vals + d_vals * e_vals
            tl.store(a_ptr + current_offsets, result, mask=combined_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    # Step 1: Process ALL 'else' branch iterations FIRST
    s161_kernel_else[(1,)](a, b, c, d, n, BLOCK_SIZE)
    
    # Step 2: Process ALL 'if' branch iterations AFTER step 1 completes
    s161_kernel_if[(1,)](a, b, c, d, e, n, BLOCK_SIZE)