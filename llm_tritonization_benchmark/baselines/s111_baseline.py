import torch

def s111_pytorch(a, b, iterations):
    """
    TSVC s111 - Conditional store
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 1; i < LEN_1D; i += 2) {
            a[i] = a[i - 1] + b[i];
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    for _ in range(2 * iterations):
        for i in range(1, a.shape[0], 2):
            a[i] = a[i - 1] + b[i]
    
    return a