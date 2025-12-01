import torch

def s491_pytorch(a, b, c, d, ip):
    """
    PyTorch implementation of TSVC s491 - indirect assignment.

    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i] + c[i] * d[i];
        }
    }

    IMPORTANT: Must process sequentially to match C semantics.
    When there are duplicate indices in ip, last write wins.
    PyTorch's a[ip] = values has undefined behavior for duplicates.

    Arrays used: a (rw), b (r), c (r), d (r), ip (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    ip = ip.contiguous()

    # Compute the values to assign
    values = b + c * d

    # Process sequentially to ensure "last write wins" for duplicates
    for i in range(len(ip)):
        a[ip[i]] = values[i]

    return a
