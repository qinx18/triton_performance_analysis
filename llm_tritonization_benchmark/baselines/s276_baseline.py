import torch

def s276_pytorch(a, b, c, d, mid):
    """
    PyTorch implementation of TSVC s276 - conditional linear combination
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (i+1 < mid) {
                a[i] += b[i] * c[i];
            } else {
                a[i] += b[i] * d[i];
            }
        }
    }
    
    Arrays used: a (rw), b (r), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    indices = torch.arange(len(a), device=a.device)
    condition = (indices + 1) < mid
    
    update = torch.where(condition, b * c, b * d)
    a += update
    
    return a