import torch

def s314_pytorch(a, x):
    """
    Find maximum value in array.
    
    Original C code:
    for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > x) {
                x = a[i];
            }
        }
    }
    
    Arrays used: a (r), x (r)
    """
    a = a.contiguous()
    x = x.contiguous()
    
    # Find maximum value in array a
    max_val = torch.max(a)
    x = torch.full_like(x, max_val)
    
    return x