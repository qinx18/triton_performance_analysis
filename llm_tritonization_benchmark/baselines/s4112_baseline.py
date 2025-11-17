import torch

def s4112_pytorch(a, b, ip, s):
    """
    PyTorch implementation of TSVC s4112 - indirect addressing with scaling.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[ip[i]] * s;
        }
    }
    
    Arrays: a (rw), b (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    ip = ip.contiguous()
    
    # Indirect addressing: gather values from b using indices in ip
    gathered_b = b[ip]
    
    # Update a with scaled gathered values
    a += gathered_b * s
    
    return a