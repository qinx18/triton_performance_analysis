import torch

def s332_pytorch(a, t):
    """
    PyTorch implementation of TSVC s332 - first value greater than threshold.
    
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
    
    Args:
        a: Input tensor (read-only)
        t: Threshold scalar value
        
    Returns:
        chksum: Sum of first value greater than threshold and its index
    """
    a = a.contiguous()
    
    # Initialize default values
    index = -2
    value = -1.0
    
    # Find first element greater than threshold
    mask = a > t
    if torch.any(mask):
        # Get the index of the first True value
        first_idx = torch.argmax(mask.float())
        index = first_idx.item()
        value = a[index].item()
    
    chksum = value + float(index)
    return chksum