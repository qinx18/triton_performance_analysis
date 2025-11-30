import torch

def s424_pytorch(a, flat_2d_array, xx):
    """
    PyTorch implementation of TSVC s424
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D - 1; i++) {
            xx[i+1] = flat_2d_array[i] + a[i];
        }
    }
    
    Arrays: a (r), flat_2d_array (r), xx (rw)
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    n = len(a)
    if n > 0:
        xx[1:n] = flat_2d_array[:n-1] + a[:n-1]