import torch

def s314_pytorch(a):
    """
    TSVC s314 - find maximum value in array
    
    Original C code:
    for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > x) {
                x = a[i];
            }
        }
    }
    
    Arrays: a (read only)
    """
    a = a.contiguous()
    
    # Find maximum value in array (equivalent to the inner loop)
    x = torch.max(a)
    
    return x