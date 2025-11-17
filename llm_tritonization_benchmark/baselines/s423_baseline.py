import torch

def s423_pytorch(a, flat_2d_array, xx):
    """
    PyTorch implementation of TSVC s423 function.
    
    Original C code:
    for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D - 1; i++) {
            flat_2d_array[i+1] = xx[i] + a[i];
        }
    }
    
    Arrays: a (r), flat_2d_array (rw), xx (r)
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    flat_2d_array[1:] = xx[:-1] + a[:-1]
    
    return flat_2d_array