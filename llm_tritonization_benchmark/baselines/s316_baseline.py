import torch

def s316_pytorch(a, x):
    """
    TSVC s316 - finding minimum value
    
    Original C code:
    for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 1; i < LEN_1D; ++i) {
            if (a[i] < x) {
                x = a[i];
            }
        }
    }
    
    Arrays: a (read), x (read-write)
    """
    a = a.contiguous()
    x = x.contiguous()
    
    # Find minimum value in array a
    x[0] = torch.min(a)
    
    return x