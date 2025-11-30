import torch

def s3251_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s3251.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++){
            a[i+1] = b[i]+c[i];
            b[i]   = c[i]*e[i];
            d[i]   = a[i]*e[i];
        }
    }
    
    Arrays: a (rw), b (rw), c (r), d (rw), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    LEN_1D = a.shape[0]
    
    # Process elements from 0 to LEN_1D-2 (since i < LEN_1D-1)
    i_range = torch.arange(LEN_1D - 1, device=a.device)
    
    # Store original a[i] values before modification for d[i] computation
    a_orig = a[i_range].clone()
    
    # a[i+1] = b[i] + c[i]
    a[1:LEN_1D] = b[:LEN_1D-1] + c[:LEN_1D-1]
    
    # b[i] = c[i] * e[i]
    b[:LEN_1D-1] = c[:LEN_1D-1] * e[:LEN_1D-1]
    
    # d[i] = a[i] * e[i] (using original a[i] values)
    d[:LEN_1D-1] = a_orig * e[:LEN_1D-1]