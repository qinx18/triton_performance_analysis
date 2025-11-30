import torch

def s317_pytorch():
    """
    TSVC s317 - scalar recurrence
    
    Original C code:
    for (int nl = 0; nl < 5*iterations; nl++) {
        q = (real_t)1.;
        for (int i = 0; i < LEN_1D/2; i++) {
            q *= (real_t).99;
        }
    }
    """
    LEN_1D = 32000
    
    q = torch.tensor(1.0)
    for i in range(LEN_1D // 2):
        q *= 0.99
    
    return q