import torch

def s241_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s241 function.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] * c[i  ] * d[i];
            b[i] = a[i] * a[i+1] * d[i];
        }
    }
    
    Arrays used: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = len(a) - 1
    
    for i in range(n):
        a[i] = b[i] * c[i] * d[i]
        b[i] = a[i] * a[i+1] * d[i]
    
    return a, b