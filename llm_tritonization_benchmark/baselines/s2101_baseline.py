import torch

def s2101_pytorch(aa, bb, cc):
    """
    PyTorch implementation of TSVC s2101 kernel.
    
    Original C code:
    for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            aa[i][i] += bb[i][i] * cc[i][i];
        }
    }
    
    Arrays: aa (rw), bb (r), cc (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    # Extract diagonal elements and perform computation
    diag_indices = torch.arange(aa.size(0), device=aa.device)
    aa[diag_indices, diag_indices] += bb[diag_indices, diag_indices] * cc[diag_indices, diag_indices]