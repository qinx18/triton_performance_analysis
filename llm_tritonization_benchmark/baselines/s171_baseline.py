import torch

def s171_pytorch(a, b, inc):
    """
    TSVC function s171 - PyTorch implementation
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i * inc] += b[i];
        }
    }
    
    Arrays used: a (rw), b (r)
    Scalar parameters: inc
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = b.size(0)
    
    for i in range(LEN_1D):
        a[i * inc] += b[i]
    
    return a