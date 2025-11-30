import torch

def s351_pytorch(a, b, alpha):
    """
    PyTorch implementation of TSVC s351 - Loop unrolling
    
    Original C code:
    for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i += 5) {
            a[i] += alpha * b[i];
            a[i + 1] += alpha * b[i + 1];
            a[i + 2] += alpha * b[i + 2];
            a[i + 3] += alpha * b[i + 3];
            a[i + 4] += alpha * b[i + 4];
        }
    }
    
    Arrays: a (rw), b (r)
    Scalar: alpha
    """
    a = a.contiguous()
    b = b.contiguous()
    
    # Handle the main loop with step size 5
    len_1d = a.size(0)
    
    # Process elements in groups of 5
    for i in range(0, len_1d - 4, 5):
        a[i] += alpha * b[i]
        a[i + 1] += alpha * b[i + 1]
        a[i + 2] += alpha * b[i + 2]
        a[i + 3] += alpha * b[i + 3]
        a[i + 4] += alpha * b[i + 4]