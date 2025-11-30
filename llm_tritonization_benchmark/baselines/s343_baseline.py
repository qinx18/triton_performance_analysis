import torch

def s343_pytorch(aa, bb, flat_2d_array):
    """
    PyTorch implementation of TSVC s343 - conditional array packing
    
    Original C code:
    for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        k = -1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (bb[j][i] > (real_t)0.) {
                    k++;
                    flat_2d_array[k] = aa[j][i];
                }
            }
        }
    }
    
    Arrays: aa (r), bb (r), flat_2d_array (rw)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = aa.shape[0]
    
    k = -1
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            if bb[j, i] > 0.0:
                k += 1
                flat_2d_array[k] = aa[j, i]