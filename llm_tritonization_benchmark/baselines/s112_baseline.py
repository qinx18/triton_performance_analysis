import torch

def s112_pytorch(a, b, iterations):
    """
    PyTorch implementation of TSVC s112 - Forward linear recurrence.
    
    Original C code:
    for (int nl = 0; nl < 3*iterations; nl++) {
        for (int i = LEN_1D - 2; i >= 0; i--) {
            a[i+1] = a[i] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    for nl in range(3 * iterations):
        for i in range(len(a) - 2, -1, -1):
            a[i + 1] = a[i] + b[i]
    
    return a