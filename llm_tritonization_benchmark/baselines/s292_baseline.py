import torch

def s292_pytorch(a, b, im1, im2):
    """
    TSVC function s292 - sliding window computation with wraparound indices
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        im1 = LEN_1D-1;
        im2 = LEN_1D-2;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + b[im1] + b[im2]) * (real_t).333;
            im2 = im1;
            im1 = i;
        }
    }
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = b.size(0)
    im1_val = LEN_1D - 1
    im2_val = LEN_1D - 2
    
    for i in range(LEN_1D):
        a[i] = (b[i] + b[im1_val] + b[im2_val]) * 0.333
        im2_val = im1_val
        im1_val = i
    
    return a