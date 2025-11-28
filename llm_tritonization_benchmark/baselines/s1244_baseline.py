import torch

def s1244_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s1244
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
            d[i] = a[i] + a[i+1];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (rw)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = a.size(0) - 1
    
    # a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a[:n] = b[:n] + c[:n] * c[:n] + b[:n] * b[:n] + c[:n]
    
    # d[i] = a[i] + a[i+1]
    d[:n] = a[:n] + a[1:n+1]