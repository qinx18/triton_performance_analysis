import torch

def s311_pytorch(a):
    """
    TSVC s311 - Sum of array elements
    
    Original C code:
    for (int nl = 0; nl < iterations*10; nl++) {
        sum = (real_t)0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
        }
    }
    
    Arrays: a (read only)
    """
    a = a.contiguous()
    
    sum_val = torch.sum(a)
    
    return sum_val