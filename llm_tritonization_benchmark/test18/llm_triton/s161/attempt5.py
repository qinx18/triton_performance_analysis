import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel_else(a_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute c[i+1] = a[i] + d[i] * d[i]
    result = a_vals + d_vals * d_vals
    
    # Store to c[i+1] - check bounds for store
    store_offsets = current_offsets + 1
    store_mask = mask & (store_offsets < (n_elements + 1))
    tl.store(c_ptr + store_offsets, result, mask=store_mask)

@triton.jit
def s161_kernel_if(a_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = c[i] + d[i] * e[i]
    result = c_vals + d_vals * e_vals
    
    # Store to a[i]
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    # Create masks for branches
    else_mask = b[:n] < 0.0
    if_mask = b[:n] >= 0.0
    
    BLOCK_SIZE = 256
    
    # Step 1: Process ALL 'else' branch iterations FIRST
    # c[i+1] = a[i] + d[i] * d[i] for all i where b[i] < 0
    if else_mask.any():
        # Create filtered tensors for else branch
        else_indices = torch.where(else_mask)[0]
        if len(else_indices) > 0:
            a_else = a[else_indices]
            c_else = c[else_indices + 1]
            d_else = d[else_indices]
            
            # Use PyTorch operation for else branch
            c[else_indices + 1] = a_else + d_else * d_else
    
    # Step 2: Process ALL 'if' branch iterations AFTER step 1 completes  
    # a[i] = c[i] + d[i] * e[i] for all i where b[i] >= 0
    if if_mask.any():
        # Create filtered tensors for if branch
        if_indices = torch.where(if_mask)[0]
        if len(if_indices) > 0:
            c_if = c[if_indices]
            d_if = d[if_indices]
            e_if = e[if_indices]
            
            # Use PyTorch operation for if branch
            a[if_indices] = c_if + d_if * e_if