import torch

def s1279_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s1279 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] < (real_t)0.) {
                if (b[i] > a[i]) {
                    c[i] += d[i] * e[i];
                }
            }
        }
    }
    
    Arrays: a (r), b (r), c (rw), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Create condition masks
    mask1 = a < 0.0
    mask2 = b > a
    combined_mask = mask1 & mask2
    
    # Update c where both conditions are true
    c[:] = torch.where(combined_mask, c + d * e, c)