import torch
import triton
import triton.language as tl

def s161_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    idx = torch.arange(n, device=a.device)
    
    # Step 1: Process ALL 'else' branch iterations FIRST
    else_mask = b[:n] < 0.0
    # c[i+1] = a[i] + d[i] * d[i] for all i where b[i] < 0
    c[1:n+1] = torch.where(else_mask, a[:n] + d[:n] * d[:n], c[1:n+1])
    
    # Step 2: Process ALL 'if' branch iterations AFTER step 1 completes
    if_mask = b[:n] >= 0.0
    # a[i] = c[i] + d[i] * e[i] for all i where b[i] >= 0
    a[:n] = torch.where(if_mask, c[:n] + d[:n] * e[:n], a[:n])