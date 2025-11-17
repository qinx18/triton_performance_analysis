import torch

def s211_pytorch(a, b, c, d, e):
    """
    TSVC function s211 PyTorch implementation.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = b[i + 1] - e[i] * d[i];
        }
    }
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Store original b values for the update
    b_orig = b.clone()
    
    # Update arrays for indices 1 to LEN_1D-2
    indices = torch.arange(1, len(a) - 1, device=a.device)
    
    a[indices] = b_orig[indices - 1] + c[indices] * d[indices]
    b[indices] = b_orig[indices + 1] - e[indices] * d[indices]
    
    return a, b