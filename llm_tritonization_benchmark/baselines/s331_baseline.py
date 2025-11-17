import torch

def s331_pytorch(a):
    """
    PyTorch implementation of TSVC s331 function.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] < (real_t)0.) {
                j = i;
            }
        }
        chksum = (real_t) j;
    }
    
    Arrays used: a (r)
    """
    a = a.contiguous()
    
    # Find all indices where a[i] < 0
    mask = a < 0.0
    indices = torch.arange(len(a), device=a.device, dtype=torch.long)
    
    # Get the last (maximum) index where condition is true, or -1 if none
    if mask.any():
        j = indices[mask].max().item()
    else:
        j = -1
    
    return a