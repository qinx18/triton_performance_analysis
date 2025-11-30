import torch

def s2244_pytorch(a, b, c, e):
    """
    PyTorch implementation of TSVC s2244
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i+1] = b[i] + e[i];
            a[i] = b[i] + c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    e = e.contiguous()
    
    n = a.size(0)
    
    for i in range(n - 1):
        a[i + 1] = b[i] + e[i]
        a[i] = b[i] + c[i]