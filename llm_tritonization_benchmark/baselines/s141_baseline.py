import torch

def s141_pytorch(bb, flat_2d_array):
    """
    PyTorch implementation of TSVC s141 kernel.
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1;
            for (int j = i; j < LEN_2D; j++) {
                flat_2d_array[k] += bb[j][i];
                k += j+1;
            }
        }
    }
    
    Args:
        bb: 2D tensor (read-only)
        flat_2d_array: 1D tensor (read-write)
    """
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = bb.size(0)
    
    for i in range(LEN_2D):
        k = (i+1) * ((i+1) - 1) // 2 + (i+1) - 1
        for j in range(i, LEN_2D):
            flat_2d_array[k] += bb[j, i]
            k += j + 1