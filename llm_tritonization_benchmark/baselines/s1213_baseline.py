import torch

def s1213_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s1213
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i-1]+c[i];
            b[i] = a[i+1]*d[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n = a.shape[0]
    
    for i in range(1, n-1):
        a[i] = b[i-1] + c[i]
        b[i] = a[i+1] * d[i]