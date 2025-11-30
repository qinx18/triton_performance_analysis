import torch

def s442_pytorch(a, b, c, d, e, indx):
    """
    PyTorch implementation of TSVC s442 - switch statement with goto labels
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            switch (indx[i]) {
                case 1:  goto L15;
                case 2:  goto L20;
                case 3:  goto L30;
                case 4:  goto L40;
            }
    L15:
            a[i] += b[i] * b[i];
            goto L50;
    L20:
            a[i] += c[i] * c[i];
            goto L50;
    L30:
            a[i] += d[i] * d[i];
            goto L50;
    L40:
            a[i] += e[i] * e[i];
    L50:
            ;
        }
    }
    
    Arrays: a (rw), b (r), c (r), d (r), e (r), indx (r)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    indx = indx.contiguous()
    
    # Case 1: a[i] += b[i] * b[i]
    mask1 = indx == 1
    a[:] = torch.where(mask1, a + b * b, a)
    
    # Case 2: a[i] += c[i] * c[i]
    mask2 = indx == 2
    a[:] = torch.where(mask2, a + c * c, a)
    
    # Case 3: a[i] += d[i] * d[i]
    mask3 = indx == 3
    a[:] = torch.where(mask3, a + d * d, a)
    
    # Case 4: a[i] += e[i] * e[i]
    mask4 = indx == 4
    a[:] = torch.where(mask4, a + e * e, a)