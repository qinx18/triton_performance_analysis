import torch

def s2102_pytorch(aa):
    """
    PyTorch implementation of TSVC s2102 function.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[j][i] = (real_t)0.;
            }
            aa[i][i] = (real_t)1.;
        }
    }
    
    Arrays used: aa (r)
    """
    aa = aa.contiguous()
    LEN_2D = aa.size(0)
    
    # Zero out all elements
    aa.fill_(0.0)
    
    # Set diagonal elements to 1
    for i in range(LEN_2D):
        aa[i, i] = 1.0
    
    return aa