import torch

def s1281_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s1281
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            x = b[i]*c[i] + a[i]*d[i] + e[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    x = b * c + a * d + e
    a[:] = x - 1.0
    b[:] = x