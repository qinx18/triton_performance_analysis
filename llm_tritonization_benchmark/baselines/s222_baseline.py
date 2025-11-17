import torch

def s222_pytorch(a, b, c, e):
    """
    PyTorch implementation of TSVC s222 function.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += b[i] * c[i];
            e[i] = e[i - 1] * e[i - 1];
            a[i] -= b[i] * c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), e (rw)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    e = e.contiguous()
    
    for i in range(1, len(a)):
        a[i] += b[i] * c[i]
        e[i] = e[i - 1] * e[i - 1]
        a[i] -= b[i] * c[i]
    
    return a, e