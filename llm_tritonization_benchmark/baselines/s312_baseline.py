import torch

def s312_pytorch(a):
    """
    PyTorch implementation of TSVC s312 - product of array elements.
    
    Original C code:
    for (int nl = 0; nl < 10*iterations; nl++) {
        prod = (real_t)1.;
        for (int i = 0; i < LEN_1D; i++) {
            prod *= a[i];
        }
    }
    
    Arrays used: a (r)
    """
    a = a.contiguous()
    prod = torch.prod(a)
    return a