import torch

def s323_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s323 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] = b[i-1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Perform one iteration of the outer loop (iterations/2 is for timing only)
    for i in range(1, len(a)):
        a[i] = b[i-1] + c[i] * d[i]
        b[i] = a[i] + c[i] * e[i]