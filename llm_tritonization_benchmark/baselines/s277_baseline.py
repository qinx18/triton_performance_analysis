import torch

def s277_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s277 kernel.

    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
                if (a[i] >= (real_t)0.) {
                    goto L20;
                }
                if (b[i] >= (real_t)0.) {
                    goto L30;
                }
                a[i] += c[i] * d[i];
    L30:
                b[i+1] = c[i] + d[i] * e[i];
    L20:
    ;
        }
    }

    Arrays: a (rw), b (rw), c (r), d (r), e (r)

    NOTE: This kernel has a loop-carried dependency on b:
    - Iteration i writes b[i+1]
    - Iteration i+1 reads b[i+1] in the condition check
    So the condition at iteration i+1 depends on whether iteration i wrote to b[i+1].
    This CANNOT be parallelized - must be sequential.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()

    n = a.size(0) - 1

    # Must process sequentially due to loop-carried dependency on b
    for i in range(n):
        if a[i].item() >= 0.0:
            continue  # goto L20 - skip everything
        if b[i].item() >= 0.0:
            # goto L30 - skip a update, but still update b
            b[i+1] = c[i] + d[i] * e[i]
            continue
        # Both conditions false: update both a and b
        a[i] = a[i] + c[i] * d[i]
        b[i+1] = c[i] + d[i] * e[i]