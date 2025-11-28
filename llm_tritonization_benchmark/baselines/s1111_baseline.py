import torch

def s1111_pytorch(a, b, c, d):
    """
    PyTorch implementation of TSVC s1111 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    len_half = a.shape[0] // 2
    
    indices = torch.arange(len_half, device=a.device)
    a[2*indices] = c[indices] * b[indices] + d[indices] * b[indices] + c[indices] * c[indices] + d[indices] * b[indices] + d[indices] * c[indices]