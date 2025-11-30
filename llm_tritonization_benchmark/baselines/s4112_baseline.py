import torch

def s4112_pytorch(a, b, ip, s):
    """
    PyTorch implementation of TSVC s4112 - indirect addressing with scalar multiplication.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[ip[i]] * s;
        }
    }
    
    Arrays: a (rw), b (r), ip (r)
    Scalar: s
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    a[:] += b[ip] * s