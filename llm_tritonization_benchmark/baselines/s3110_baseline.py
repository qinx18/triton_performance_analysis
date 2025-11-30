import torch

def s3110_pytorch(aa):
    """
    PyTorch implementation of TSVC s3110 - finding maximum value and its indices in 2D array
    
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
    
    Arrays: aa (r)
    """
    aa = aa.contiguous()
    
    # Find maximum value and its linear index
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    # Convert linear index to 2D indices
    LEN_2D = aa.shape[0]
    xindex = max_idx // LEN_2D
    yindex = max_idx % LEN_2D
    
    # Calculate checksum (equivalent to C code behavior)
    chksum = max_val + xindex.float() + yindex.float()
    
    return chksum