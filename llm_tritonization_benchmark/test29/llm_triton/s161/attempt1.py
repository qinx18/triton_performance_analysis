import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Create masks for branches
        else_mask = (b_vals < 0.0) & mask
        if_mask = (b_vals >= 0.0) & mask
        
        # Process else branch: c[i+1] = a[i] + d[i] * d[i]
        if tl.any(else_mask):
            a_vals = tl.load(a_ptr + current_offsets, mask=else_mask)
            else_result = a_vals + d_vals * d_vals
            # Write to c[i+1]
            next_offsets = current_offsets + 1
            next_mask = else_mask & (next_offsets < (n + 1))
            tl.store(c_ptr + next_offsets, else_result, mask=next_mask)
        
        # Process if branch: a[i] = c[i] + d[i] * e[i]
        if tl.any(if_mask):
            c_vals = tl.load(c_ptr + current_offsets, mask=if_mask)
            e_vals = tl.load(e_ptr + current_offsets, mask=if_mask)
            if_result = c_vals + d_vals * e_vals
            tl.store(a_ptr + current_offsets, if_result, mask=if_mask)

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    # First pass: Process else branch completely
    idx = torch.arange(n, device=a.device)
    else_mask = b[:n] < 0.0
    if else_mask.any():
        c[1:n+1] = torch.where(else_mask, a[:n] + d[:n] * d[:n], c[1:n+1])
    
    # Second pass: Process if branch completely  
    if_mask = b[:n] >= 0.0
    if if_mask.any():
        a[:n] = torch.where(if_mask, c[:n] + d[:n] * e[:n], a[:n])