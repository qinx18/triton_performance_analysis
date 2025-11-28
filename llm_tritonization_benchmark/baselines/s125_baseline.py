import torch

def s125_pytorch(aa, bb, cc, flat_2d_array):
    """
    PyTorch implementation of TSVC s125 kernel.
    
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
    
    Arrays: aa (r), bb (r), cc (r), flat_2d_array (rw)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    # Flatten the 2D arrays and compute the operation
    aa_flat = aa.flatten()
    bb_flat = bb.flatten()
    cc_flat = cc.flatten()
    
    # Perform the computation: aa[i][j] + bb[i][j] * cc[i][j]
    flat_2d_array[:] = aa_flat + bb_flat * cc_flat