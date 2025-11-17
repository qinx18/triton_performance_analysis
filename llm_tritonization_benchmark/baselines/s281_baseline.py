import torch

def s281_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s281
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            x = a[LEN_1D-i-1] + b[i] * c[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
    }
    
    Arrays: a (rw), b (rw), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    LEN_1D = a.shape[0]
    
    # Create indices for reversed access to a
    reverse_indices = torch.arange(LEN_1D - 1, -1, -1, device=a.device)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x = a[reverse_indices] + b * c
    
    # Update arrays
    a[:] = x - 1.0
    b[:] = x
    
    return a, b