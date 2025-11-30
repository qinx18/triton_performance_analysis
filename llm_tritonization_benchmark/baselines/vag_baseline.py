import torch

def vag_pytorch(a, b, ip):
    """
    PyTorch implementation of TSVC vag function.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[ip[i]];
        }
    }
    
    Arrays: a (rw), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    a[:] = b[ip]