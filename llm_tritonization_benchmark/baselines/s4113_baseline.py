import torch

def s4113_pytorch(a, b, c, ip):
    """
    PyTorch implementation of TSVC s4113 - indirect addressing with computation
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[ip[i]] + c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    ip = ip.contiguous()
    
    # Convert ip to long for indexing
    indices = ip.long()
    
    # Perform indirect addressing computation
    a[indices] = b[indices] + c