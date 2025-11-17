import torch

def s277_pytorch(a, b, c, d, e):
    """
    PyTorch implementation of TSVC s277 - conditional assignments with goto statements.
    
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
    
    Args:
        a: read-write tensor
        b: read-write tensor  
        c: read-only tensor
        d: read-only tensor
        e: read-only tensor
    
    Returns:
        tuple: (modified_a, modified_b)
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    LEN_1D = a.shape[0]
    
    # Create slices for i from 0 to LEN_1D-2 (since loop goes to LEN_1D-1)
    i_range = torch.arange(LEN_1D - 1, device=a.device)
    
    # Conditions
    a_nonneg = a[i_range] >= 0.0  # if a[i] >= 0, goto L20 (skip everything)
    b_nonneg = b[i_range] >= 0.0  # if b[i] >= 0, goto L30 (skip a[i] update)
    
    # Update a[i] only when both a[i] < 0 and b[i] < 0
    update_a_mask = ~a_nonneg & ~b_nonneg
    a[i_range] = torch.where(update_a_mask, 
                            a[i_range] + c[i_range] * d[i_range], 
                            a[i_range])
    
    # Update b[i+1] when a[i] < 0 (not goto L20)
    update_b_mask = ~a_nonneg
    b[i_range + 1] = torch.where(update_b_mask, 
                                c[i_range] + d[i_range] * e[i_range], 
                                b[i_range + 1])
    
    return a, b