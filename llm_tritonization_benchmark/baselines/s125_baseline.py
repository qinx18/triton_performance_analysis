import torch

def s125_pytorch(aa, bb, cc, flat_2d_array):
    """
    PyTorch implementation of TSVC s125 - 2D to 1D array flattening with computation.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        k = -1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                k++;
                flat_2d_array[k] = aa[i][j] + bb[i][j] * cc[i][j];
            }
        }
    }
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    # Compute aa + bb * cc for all elements
    result = aa + bb * cc
    
    # Flatten the result to 1D array
    flat_2d_array[:] = result.flatten()
    
    return flat_2d_array