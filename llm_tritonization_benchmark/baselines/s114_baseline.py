import torch

def s114_pytorch(aa, bb):
    """
    PyTorch implementation of TSVC s114 - lower triangular matrix update.
    
    Original C code:
    for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < i; j++) {
                aa[i][j] = aa[j][i] + bb[i][j];
            }
        }
    }
    
    Arrays: aa (rw), bb (r)
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Create index matrices
    i_indices = torch.arange(LEN_2D, device=aa.device).unsqueeze(1)
    j_indices = torch.arange(LEN_2D, device=aa.device).unsqueeze(0)
    
    # Create mask for j < i (lower triangular)
    mask = j_indices < i_indices
    
    # Update aa where mask is True
    aa = torch.where(mask, aa.transpose(0, 1) + bb, aa)
    
    return aa