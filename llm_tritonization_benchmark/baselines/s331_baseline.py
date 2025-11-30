import torch

def s331_pytorch(a):
    """
    PyTorch implementation of TSVC s331 - find last index where array element is negative
    
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
    
    Arrays: a (read-only)
    """
    a = a.contiguous()
    
    # Find all indices where a < 0
    mask = a < 0.0
    
    # If no negative elements, j should be -1
    if not torch.any(mask):
        j = torch.tensor(-1, dtype=torch.int32, device=a.device)
    else:
        # Find the last index where condition is true
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        j = indices[-1]
    
    # The chksum would be j, but since we don't modify any arrays,
    # we just need to perform the computation for equivalence
    chksum = j.float()
    
    return chksum