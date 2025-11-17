import torch

def s272_pytorch(a, b, c, d, e, t):
    """
    PyTorch implementation of TSVC s272 - conditional updates with arithmetic operations.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (e[i] >= t) {
                a[i] += c[i] * d[i];
                b[i] += c[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r), e (r)
    Scalar parameters: t
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    mask = e >= t
    a_update = c * d
    b_update = c * c
    
    a = torch.where(mask, a + a_update, a)
    b = torch.where(mask, b + b_update, b)
    
    return a, b