import torch

def s422_pytorch(a, flat_2d_array, xx):
    """
    PyTorch implementation of TSVC s422
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            xx[i] = flat_2d_array[i + 8] + a[i];
        }
    }
    
    Arrays: a (read), flat_2d_array (read), xx (read-write)
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    xx[:] = flat_2d_array[8:8+len(a)] + a