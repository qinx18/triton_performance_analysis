import torch

def s251_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s251.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            s = b[i] + c[i] * d[i];
            a[i] = s * s;
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    s = b + c * d
    a[:] = s * s
    
    return a