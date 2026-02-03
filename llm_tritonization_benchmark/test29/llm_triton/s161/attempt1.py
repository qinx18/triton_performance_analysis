import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Branch condition
    else_mask = b_vals < 0.0
    
    # Compute else branch: c[i+1] = a[i] + d[i] * d[i]
    else_result = a_vals + d_vals * d_vals
    
    # Store to c[i+1] for else branch
    store_mask_else = mask & else_mask
    store_offsets_c = offsets + 1
    store_mask_c_bounds = store_offsets_c < (n_elements + 1)
    final_store_mask_else = store_mask_else & store_mask_c_bounds
    tl.store(c_ptr + store_offsets_c, else_result, mask=final_store_mask_else)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    # Step 1: Process else branch first
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s161_kernel[grid](a, b, c, d, e, n, BLOCK_SIZE)
    
    # Step 2: Process if branch after synchronization
    idx = torch.arange(n, device=a.device)
    if_mask = b[:n] >= 0.0
    if torch.any(if_mask):
        a[:n][if_mask] = (c[:n] + d[:n] * e[:n])[if_mask]