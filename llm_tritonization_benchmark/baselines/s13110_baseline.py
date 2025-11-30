import torch

def s13110_pytorch(aa):
    """
    PyTorch implementation of TSVC s13110 - finding maximum element and its indices in a 2D array.
    
    Original C code:
    for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        max = aa[(0)][0];
        xindex = 0;
        yindex = 0;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (aa[i][j] > max) {
                    max = aa[i][j];
                    xindex = i;
                    yindex = j;
                }
            }
        }
        chksum = max + (real_t) xindex + (real_t) yindex;
    }
    
    Arrays: aa (read-only)
    """
    aa = aa.contiguous()
    
    # Find the maximum value and its linear index
    max_val = torch.max(aa)
    max_indices = torch.argmax(aa.flatten())
    
    # Convert linear index back to 2D coordinates
    len_2d = aa.shape[0]
    xindex = max_indices // len_2d
    yindex = max_indices % len_2d
    
    # Calculate checksum (equivalent to the C code's final computation)
    chksum = max_val + xindex.float() + yindex.float()
    
    return chksum