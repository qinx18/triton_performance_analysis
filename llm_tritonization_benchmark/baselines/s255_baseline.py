import torch

def s255_pytorch(a, b):
    """
    PyTorch implementation of TSVC s255 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        x = b[LEN_1D-1];
        y = b[LEN_1D-2];
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + x + y) * (real_t).333;
            y = x;
            x = b[i];
        }
    }
    
    Args:
        a: torch.Tensor (read-write)
        b: torch.Tensor (read-only)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = b.shape[0]
    
    x = b[LEN_1D-1].clone()
    y = b[LEN_1D-2].clone()
    
    for i in range(LEN_1D):
        a[i] = (b[i] + x + y) * 0.333
        y = x
        x = b[i]