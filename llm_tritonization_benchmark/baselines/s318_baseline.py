import torch

def s318_pytorch(a, inc):
    """
    PyTorch implementation of TSVC s318 - finding maximum absolute value and its index.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        k = 0;
        index = 0;
        max = ABS(a[0]);
        k += inc;
        for (int i = 1; i < LEN_1D; i++) {
            if (ABS(a[k]) <= max) {
                goto L5;
            }
            index = i;
            max = ABS(a[k]);
    L5:
            k += inc;
        }
        chksum = max + (real_t) index;
    }
    
    Arrays: a (read-only)
    """
    a = a.contiguous()
    
    k = 0
    index = 0
    max_val = torch.abs(a[0])
    k += inc
    
    for i in range(1, len(a)):
        if k >= len(a):
            break
        current_abs = torch.abs(a[k])
        if current_abs > max_val:
            index = i
            max_val = current_abs
        k += inc
    
    chksum = max_val + index
    return chksum