import torch

def s123_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s123 - conditional assignment with variable indexing.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < (LEN_1D/2); i++) {
            j++;
            a[j] = b[i] + d[i] * e[i];
            if (c[i] > (real_t)0.) {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r), e (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    len_1d = b.shape[0] * 2  # Since we iterate over LEN_1D/2, b.shape[0] represents LEN_1D/2
    half_len = len_1d // 2
    
    # Reset the output array portion we'll be writing to
    a[:len_1d] = 0
    
    j = 0
    for i in range(half_len):
        # First assignment: a[j] = b[i] + d[i] * e[i]
        a[j] = b[i] + d[i] * e[i]
        j += 1
        
        # Conditional assignment
        if c[i] > 0.0:
            a[j] = c[i] + d[i] * e[i]
            j += 1
    
    return a