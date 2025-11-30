import torch

def s453_pytorch(a, b):
    """
    PyTorch implementation of TSVC s453.
    
    Original C code:
    for (int nl = 0; nl < iterations*2; nl++) {
        s = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            s += (real_t)2.;
            a[i] = s * b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    s = 0.0
    for i in range(len(a)):
        s += 2.0
        a[i] = s * b[i]