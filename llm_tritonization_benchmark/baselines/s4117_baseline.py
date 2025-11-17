import torch

def s4117_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s4117
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + c[i/2] * d[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    LEN_1D = a.size(0)
    indices = torch.arange(LEN_1D, device=a.device)
    c_indices = indices // 2
    
    a[:] = b[:] + c[c_indices] * d[:]
    
    return a