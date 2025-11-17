import torch

def s332_pytorch(a, t):
    """
    PyTorch implementation of TSVC s332 - first value greater than threshold
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        index = -2;
        value = -1.;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > t) {
                index = i;
                value = a[i];
                goto L20;
            }
        }
    L20:
        chksum = value + (real_t) index;
    }
    
    Arrays used: a (r)
    Scalar parameters: t
    """
    a = a.contiguous()
    
    # Find first element greater than t
    mask = a > t
    
    # If any element satisfies condition
    if torch.any(mask):
        # Find first index where condition is true
        indices = torch.arange(len(a), device=a.device)
        valid_indices = torch.where(mask, indices, torch.tensor(len(a), device=a.device))
        first_idx = torch.min(valid_indices)
        
        index = first_idx
        value = a[first_idx]
    else:
        index = torch.tensor(-2, device=a.device, dtype=torch.long)
        value = torch.tensor(-1.0, device=a.device, dtype=a.dtype)
    
    # Compute checksum (not used in return but part of original computation)
    chksum = value + index.to(a.dtype)
    
    return a