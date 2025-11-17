import torch

def s2101_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s2101 - diagonal matrix multiplication.
    
    Original C code:
    for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            aa[i][i] += bb[i][i] * cc[i][i];
        }
    }
    
    Arrays used: aa (rw), bb (r), cc (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    # Get diagonal indices
    diag_indices = torch.arange(aa.size(0), device=aa.device)
    
    # Update diagonal elements: aa[i][i] += bb[i][i] * cc[i][i]
    aa[diag_indices, diag_indices] += bb[diag_indices, diag_indices] * cc[diag_indices, diag_indices]
    
    return aa