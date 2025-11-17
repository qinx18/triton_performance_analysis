import torch

def s126_pytorch(bb, cc, flat_2d_array):
    """
    PyTorch implementation of TSVC s126 function.
    
    Original C code:
    for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        k = 1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i];
                ++k;
            }
            ++k;
        }
    }
    """
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = bb.shape[0]
    k = 1
    
    for i in range(LEN_2D):
        for j in range(1, LEN_2D):
            bb[j, i] = bb[j-1, i] + flat_2d_array[k-1] * cc[j, i]
            k += 1
        k += 1
    
    return bb