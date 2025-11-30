import torch

def s241_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s241 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] * c[i  ] * d[i];
            b[i] = a[i] * a[i+1] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = a.size(0) - 1
    
    # First statement: a[i] = b[i] * c[i] * d[i]
    a[:n] = b[:n] * c[:n] * d[:n]
    
    # Second statement: b[i] = a[i] * a[i+1] * d[i]
    b[:n] = a[:n] * a[1:n+1] * d[:n]