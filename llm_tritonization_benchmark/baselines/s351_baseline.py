import torch

def s351_pytorch(a, b, alpha):
    """
    PyTorch implementation of TSVC s351 - unrolled SAXPY operation.
    
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
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    
    # Process in chunks of 5
    for i in range(0, len_1d, 5):
        if i < len_1d:
            a[i] += alpha * b[i]
        if i + 1 < len_1d:
            a[i + 1] += alpha * b[i + 1]
        if i + 2 < len_1d:
            a[i + 2] += alpha * b[i + 2]
        if i + 3 < len_1d:
            a[i + 3] += alpha * b[i + 3]
        if i + 4 < len_1d:
            a[i + 4] += alpha * b[i + 4]
    
    return a