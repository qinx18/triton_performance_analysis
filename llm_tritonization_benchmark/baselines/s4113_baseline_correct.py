import torch

def s4113_pytorch(a, b, c, ip):
    """
    PyTorch implementation of TSVC s4113 - indirect addressing.

    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[ip[i]] + c[i];
        }
    }

    IMPORTANT: Must process sequentially to match C semantics.
    When there are duplicate indices in ip, last write wins.
    PyTorch's a[ip] = values has undefined behavior for duplicates.

    Arrays: a (rw), b (r), c (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    ip = ip.contiguous()

    # Process sequentially to ensure "last write wins" for duplicates
    for i in range(len(ip)):
        idx = ip[i]
        a[idx] = b[idx] + c[i]

    return a
