import torch

def s4113_pytorch(a, b, c, ip):
    """
    PyTorch implementation of TSVC s4113 - indirect addressing.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[ip[i]] + c[i];
        }
    }
    
    Arrays: a (r), b (r), c (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    ip = ip.contiguous()
    
    # Indirect addressing: a[ip[i]] = b[ip[i]] + c[i]
    indices = ip.long()
    a[indices] = b[indices] + c
    
    return a