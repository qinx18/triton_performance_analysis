import torch

def s279_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s279 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * d[i];
            if (b[i] <= a[i]) {
                goto L30;
            }
            c[i] += d[i] * e[i];
            goto L30;
    L20:
            c[i] = -c[i] + e[i] * e[i];
    L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Condition for first branch
    cond1 = a > 0.0
    
    # Path when a[i] <= 0
    b_temp = -b + d * d
    cond2 = b_temp <= a
    
    # Update b for path when a[i] <= 0
    b[:] = torch.where(cond1, b, b_temp)
    
    # Update c for both paths
    # When a[i] > 0: c[i] = -c[i] + e[i] * e[i]
    # When a[i] <= 0 and b[i] > a[i]: c[i] += d[i] * e[i]
    # When a[i] <= 0 and b[i] <= a[i]: c[i] unchanged
    c_path1 = -c + e * e  # a[i] > 0
    c_path2 = c + d * e   # a[i] <= 0 and b[i] > a[i]
    
    c[:] = torch.where(cond1, c_path1, torch.where(~cond2, c_path2, c))
    
    # Final update: a[i] = b[i] + c[i] * d[i]
    a[:] = b + c * d