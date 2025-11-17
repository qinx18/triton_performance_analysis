import torch

def s252_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s252 - scalar and array expansion
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        t = (real_t) 0.;
        for (int i = 0; i < LEN_1D; i++) {
            s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    t = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    for i in range(len(a)):
        s = b[i] * c[i]
        a[i] = s + t
        t = s
    
    return a