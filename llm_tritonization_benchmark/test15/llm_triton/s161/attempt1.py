import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, c_ptr, d_ptr, mask_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load mask for else branch (b[i] < 0)
        branch_mask = tl.load(mask_ptr + current_offsets, mask=mask, other=False)
        
        # Only process elements where else branch should execute
        valid_mask = mask & branch_mask
        
        # c[i+1] = a[i] + d[i] * d[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=valid_mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=valid_mask, other=0.0)
        result = a_vals + d_vals * d_vals
        
        # Store to c[i+1]
        store_offsets = current_offsets + 1
        store_mask = valid_mask & (store_offsets < (n_elements + 1))
        tl.store(c_ptr + store_offsets, result, mask=store_mask)

@triton.jit
def s161_kernel_if(a_ptr, c_ptr, d_ptr, e_ptr, mask_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load mask for if branch (b[i] >= 0)
        branch_mask = tl.load(mask_ptr + current_offsets, mask=mask, other=True)
        
        # Only process elements where if branch should execute
        valid_mask = mask & branch_mask
        
        # a[i] = c[i] + d[i] * e[i]
        c_vals = tl.load(c_ptr + current_offsets, mask=valid_mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=valid_mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=valid_mask, other=0.0)
        result = c_vals + d_vals * e_vals
        
        tl.store(a_ptr + current_offsets, result, mask=valid_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    # Create masks for both branches
    else_mask = b[:n] < 0.0
    if_mask = b[:n] >= 0.0
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n, BLOCK_SIZE)
    
    # Step 1: Process else branch first (c[i+1] = a[i] + d[i] * d[i])
    s161_kernel_else[(grid_size,)](
        a, c, d, else_mask, n, BLOCK_SIZE
    )
    
    # Step 2: Process if branch after (a[i] = c[i] + d[i] * e[i])
    s161_kernel_if[(grid_size,)](
        a, c, d, e, if_mask, n, BLOCK_SIZE
    )