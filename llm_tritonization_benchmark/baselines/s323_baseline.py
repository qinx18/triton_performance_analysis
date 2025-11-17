import torch

def s323_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s323 function.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] = b[i-1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
    }
    
    Arrays used: a (rw), b (rw), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Loop from i=1 to LEN_1D-1
    for i in range(1, len(a)):
        a[i] = b[i-1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]
    
    return a, b