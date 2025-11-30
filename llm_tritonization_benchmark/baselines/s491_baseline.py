import torch

def s491_pytorch(a, b, c, d, ip):
    """
    PyTorch implementation of TSVC s491 - indirect assignment with computation.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i] + c[i] * d[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()
    
    # Compute values and assign to indirect indices
    values = b + c * d
    a[ip] = values