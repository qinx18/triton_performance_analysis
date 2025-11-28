import torch

def s124_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s124 kernel.
    
    Original C code:
    for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i] + d[i] * e[i];
            } else {
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
    
    # Compute d[i] * e[i] for all elements
    de_product = d * e
    
    # Use torch.where to select between b[i] and c[i] based on condition
    selected_values = torch.where(b > 0.0, b, c)
    
    # Compute final result
    result = selected_values + de_product
    
    # Copy result to array a in-place
    a[:] = result