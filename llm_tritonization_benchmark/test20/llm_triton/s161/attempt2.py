import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(c_ptr, a_ptr, d_ptr, mask_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load mask values (1 for b[i] < 0, 0 otherwise)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Only process where b[i] < 0 (mask_vals == 1)
    else_mask = mask & (mask_vals == 1)
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=else_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=else_mask, other=0.0)
    
    # Compute c[i+1] = a[i] + d[i] * d[i]
    result = a_vals + d_vals * d_vals
    
    # Store to c[i+1]
    tl.store(c_ptr + offsets + 1, result, mask=else_mask)

@triton.jit
def s161_kernel_if(a_ptr, c_ptr, d_ptr, e_ptr, mask_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n
    
    # Load mask values (1 for b[i] < 0, 0 otherwise)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=0)
    
    # Only process where b[i] >= 0 (mask_vals == 0)
    if_mask = mask & (mask_vals == 0)
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=if_mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=if_mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=if_mask, other=0.0)
    
    # Compute a[i] = c[i] + d[i] * e[i]
    result = c_vals + d_vals * e_vals
    
    # Store to a[i]
    tl.store(a_ptr + offsets, result, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Create mask for b[i] < 0
    else_mask = (b[:n] < 0.0).int()
    
    # Step 1: Process ALL 'else' branch iterations FIRST
    # c[i+1] = a[i] + d[i] * d[i] for all i where b[i] < 0
    s161_kernel_else[grid](
        c, a, d, else_mask, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Process ALL 'if' branch iterations AFTER step 1 completes
    # a[i] = c[i] + d[i] * e[i] for all i where b[i] >= 0
    s161_kernel_if[grid](
        a, c, d, e, else_mask, n, BLOCK_SIZE=BLOCK_SIZE
    )