import torch

def s471_pytorch(b, c, d, e, x, m):
    """
    PyTorch implementation of TSVC s471 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < m; i++) {
            x[i] = b[i] + d[i] * d[i];
            s471s();
            b[i] = c[i] + d[i] * e[i];
        }
    }
    
    Arrays: b (rw), c (r), d (r), e (r), x (rw)
    """
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    x = x.contiguous()
    
    x[:m] = b[:m] + d[:m] * d[:m]
    b[:m] = c[:m] + d[:m] * e[:m]