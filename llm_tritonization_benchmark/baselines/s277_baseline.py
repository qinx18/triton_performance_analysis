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
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n = a.size(0) - 1
    
    # Create masks for the conditional logic
    a_negative = a[:n] < 0.0
    b_negative = b[:n] < 0.0
    
    # Only update a[i] if both a[i] < 0 and b[i] < 0
    update_a_mask = a_negative & b_negative
    a[:n] = torch.where(update_a_mask, a[:n] + c[:n] * d[:n], a[:n])
    
    # Update b[i+1] if a[i] < 0 (regardless of b[i])
    update_b_mask = a_negative
    b[1:n+1] = torch.where(update_b_mask, c[:n] + d[:n] * e[:n], b[1:n+1])