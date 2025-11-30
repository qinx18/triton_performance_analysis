import torch

def s2712_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s2712 kernel.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > b[i]) {
                a[i] += b[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    mask = a > b
    a[:] = torch.where(mask, a + b * c, a)