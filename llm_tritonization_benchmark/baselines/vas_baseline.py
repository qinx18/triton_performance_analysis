import torch

def vas_pytorch(a, b, ip):
    """
    PyTorch implementation of TSVC vas function.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i];
        }
    }
    
    Arrays: a (r), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    a[ip] = b