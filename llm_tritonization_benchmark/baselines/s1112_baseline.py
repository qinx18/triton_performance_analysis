import torch

def s1112_pytorch(a, b):
    """
    PyTorch implementation of TSVC s1112 function.
    
    Original C code:
    for (int nl = 0; nl < iterations*3; nl++) {
        for (int i = LEN_1D - 1; i >= 0; i--) {
            a[i] = b[i] + (real_t) 1.;
        }
    }
    
    Arrays used: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Reverse iteration: from LEN_1D-1 down to 0
    for i in range(len(a) - 1, -1, -1):
        a[i] = b[i] + 1.0