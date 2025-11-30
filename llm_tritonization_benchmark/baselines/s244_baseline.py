import torch

def s244_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s244
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; ++i) {
            a[i] = b[i] + c[i] * d[i];
            b[i] = c[i] + b[i];
            a[i+1] = b[i] + a[i+1] * d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    len_1d = a.size(0)
    
    for i in range(len_1d - 1):
        a[i] = b[i] + c[i] * d[i]
        b[i] = c[i] + b[i]
        a[i+1] = b[i] + a[i+1] * d[i]