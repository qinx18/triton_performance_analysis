import torch

def s4114_pytorch(a, b, c, d, ip, n1):
    """
    PyTorch implementation of TSVC s4114
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = n1-1; i < LEN_1D; i++) {
            k = ip[i];
            a[i] = b[i] + c[LEN_1D-k+1-2] * d[i];
            k += 5;
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r), ip (r)
    Scalar parameters: n1
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()
    
    LEN_1D = a.shape[0]
    
    for i in range(n1-1, LEN_1D):
        k = ip[i].item()
        a[i] = b[i] + c[LEN_1D-k+1-2] * d[i]
    
    return a