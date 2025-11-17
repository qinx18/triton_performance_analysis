import torch

def vsumr_pytorch(a):
    """
    PyTorch implementation of TSVC vsumr function.
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
        }
    }
    
    Arrays used: a (r)
    """
    a = a.contiguous()
    
    sum_val = torch.sum(a)
    
    return a