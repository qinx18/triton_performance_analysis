import torch

def s441_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s441 - conditional linear recurrence
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] < (real_t)0.) {
                a[i] += b[i] * c[i];
            } else if (d[i] == (real_t)0.) {
                a[i] += b[i] * b[i];
            } else {
                a[i] += c[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (rw)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    # Create masks for the three conditions
    mask_negative = d < 0.0
    mask_zero = d == 0.0
    mask_positive = ~(mask_negative | mask_zero)
    
    # Apply conditional updates
    a = torch.where(mask_negative, a + b * c, a)
    a = torch.where(mask_zero, a + b * b, a)
    a = torch.where(mask_positive, a + c * c, a)
    
    return a