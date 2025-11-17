import torch

def s121_pytorch(a, b):
    """
    PyTorch implementation of TSVC s121 function.
    
    Original C code:
    for (int nl = 0; nl < 3*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            j = i + 1;
            a[i] = a[j] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # For i = 0 to LEN_1D-2, a[i] = a[i+1] + b[i]
    if len(a) > 1:
        a[:-1] = a[1:] + b[:-1]
    
    return a