import torch

def s151_pytorch(a, b):
    """
    PyTorch implementation of TSVC s151.
    
    Original C code:
    for (int nl = 0; nl < 5*iterations; nl++) {
        s151s(a, b,  1);
    }
    
    Arrays: a (read), b (read)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # s151s with step=1 performs: a[i] = (a[i] + b[i]) / 2
    a[:] = (a + b) / 2.0