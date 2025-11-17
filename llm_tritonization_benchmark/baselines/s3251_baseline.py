import torch

def s3251_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s3251.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++){
            a[i+1] = b[i]+c[i];
            b[i]   = c[i]*e[i];
            d[i]   = a[i]*e[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (rw), e (r)
    """
    a = a.contiguous().clone()
    b = b.contiguous().clone()
    d = d.contiguous().clone()
    
    LEN_1D = len(a)
    
    for i in range(LEN_1D - 1):
        a[i+1] = b[i] + c[i]
        b[i] = c[i] * e[i]
        d[i] = a[i] * e[i]
    
    return a, b, d