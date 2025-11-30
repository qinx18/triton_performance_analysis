import torch

def s162_pytorch(a, b, c, k):
    """
    PyTorch implementation of TSVC s162
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        if (k > 0) {
            for (int i = 0; i < LEN_1D-1; i++) {
                a[i] = a[i + k] + b[i] * c[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r)
    Scalar parameters: k
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    if k > 0:
        len_1d = a.shape[0]
        if len_1d > 1 and k < len_1d:
            end_idx = min(len_1d - 1, len_1d - k)
            a[:end_idx] = a[k:k+end_idx] + b[:end_idx] * c[:end_idx]