import torch

def s421_pytorch(a, xx, yy):
    """
    PyTorch implementation of TSVC s421.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        yy = xx;
        for (int i = 0; i < LEN_1D - 1; i++) {
            xx[i] = yy[i+1] + a[i];
        }
    }
    
    Arrays: a (r), xx (rw), yy (r)
    """
    a = a.contiguous()
    xx = xx.contiguous()
    yy = yy.contiguous()
    
    yy = xx.clone()
    xx[:-1] = yy[1:] + a[:-1]
    
    return xx, yy