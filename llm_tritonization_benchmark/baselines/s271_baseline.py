import torch

def s271_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s271 function.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                a[i] += b[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    mask = b > 0.0
    a = torch.where(mask, a + b * c, a)
    
    return a