import torch

def s452_pytorch(a, b, c):
    """
    PyTorch implementation of TSVC s452
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + c[i] * (real_t) (i+1);
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    indices = torch.arange(1, len(a) + 1, dtype=a.dtype, device=a.device)
    a[:] = b + c * indices
    
    return a