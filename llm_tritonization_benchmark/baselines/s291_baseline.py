import torch

def s291_pytorch(a, b):
    """
    PyTorch implementation of TSVC s291 kernel.
    
    Original C code:
    for (int nl = 0; nl < 2*iterations; nl++) {
        im1 = LEN_1D-1;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + b[im1]) * (real_t).5;
            im1 = i;
        }
    }
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = a.shape[0]
    
    # Create shifted version of b where b[im1] represents previous element
    # im1 starts at LEN_1D-1, then becomes 0, 1, 2, ...
    b_shifted = torch.cat([b[LEN_1D-1:LEN_1D], b[0:LEN_1D-1]])
    
    a[:] = (b + b_shifted) * 0.5