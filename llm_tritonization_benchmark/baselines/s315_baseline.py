import torch

def s315_pytorch(a, x):
    """
    PyTorch implementation of TSVC s315 - find maximum value and its index.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        x = a[0];
        index = 0;
        for (int i = 0; i < LEN_1D; ++i) {
            if (a[i] > x) {
                x = a[i];
                index = i;
            }
        }
        chksum = x + (real_t) index;
    }
    """
    a = a.contiguous()
    x = x.contiguous()
    
    # Find maximum value and its index
    max_val, max_idx = torch.max(a, dim=0)
    
    # Update x with the maximum value
    x[0] = max_val
    
    return a, x