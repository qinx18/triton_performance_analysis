import torch

def s275_pytorch(aa, bb, cc):
    """
    TSVC s275 - conditional inner loop with recurrence
    
    Original C code:
    for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            if (aa[0][i] > (real_t)0.) {
                for (int j = 1; j < LEN_2D; j++) {
                    aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i];
                }
            }
        }
    }
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    LEN_2D = aa.shape[0]
    
    for i in range(LEN_2D):
        if aa[0, i] > 0.0:
            for j in range(1, LEN_2D):
                aa[j, i] = aa[j-1, i] + bb[j, i] * cc[j, i]