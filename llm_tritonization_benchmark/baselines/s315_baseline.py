import torch

def s315_pytorch(a):
    """
    TSVC s315 - Find maximum element and its index.
    
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
    
    Arrays: a (read-only)
    """
    a = a.contiguous()
    
    # Find maximum value and its index
    max_val, max_idx = torch.max(a, dim=0)
    
    # Calculate checksum (equivalent to chksum = x + (real_t) index)
    chksum = max_val + max_idx.float()
    
    return chksum