import torch

def s253_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s253 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > b[i]) {
                s = a[i] - b[i] * d[i];
                c[i] += s;
                a[i] = s;
            }
        }
    }
    
    Arrays used: a (rw), b (r), c (rw), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    mask = a > b
    s = a - b * d
    
    c[:] = torch.where(mask, c + s, c)
    a[:] = torch.where(mask, s, a)