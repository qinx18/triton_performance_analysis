import torch

def s221_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s221 function.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += c[i] * d[i];
            b[i] = b[i - 1] + a[i] + d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    for i in range(1, len(a)):
        a[i] += c[i] * d[i]
        b[i] = b[i - 1] + a[i] + d[i]
    
    return a, b