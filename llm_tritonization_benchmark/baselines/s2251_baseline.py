import torch

def s2251_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s2251
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        real_t s = (real_t)0.0;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = s*e[i];
            s = b[i]+c[i];
            b[i] = a[i]+d[i];
        }
    }
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    s = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    for i in range(len(a)):
        a[i] = s * e[i]
        s = b[i] + c[i]
        b[i] = a[i] + d[i]