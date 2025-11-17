import torch

def s1279_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s1279
    
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
    
    # Create condition mask: a[i] < 0 AND b[i] > a[i]
    condition = (a < 0.0) & (b > a)
    
    # Update c where condition is true: c[i] += d[i] * e[i]
    c = torch.where(condition, c + d * e, c)
    
    return c