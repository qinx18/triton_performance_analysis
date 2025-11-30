import torch

def s292_pytorch(a, b):
    """
    PyTorch implementation of TSVC s292.
    
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
    
    Arrays: a (rw), b (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    
    LEN_1D = b.shape[0]
    
    # Create shifted versions of b for vectorized computation
    b_im1 = torch.zeros_like(b)
    b_im2 = torch.zeros_like(b)
    
    # Initial values
    im1 = LEN_1D - 1
    im2 = LEN_1D - 2
    
    # Fill the shifted arrays according to the C loop logic
    for i in range(LEN_1D):
        b_im1[i] = b[im1]
        b_im2[i] = b[im2]
        im2 = im1
        im1 = i
    
    # Vectorized computation
    a[:] = (b + b_im1 + b_im2) * 0.333