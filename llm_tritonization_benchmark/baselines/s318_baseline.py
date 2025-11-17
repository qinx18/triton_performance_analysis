import torch

def s318_pytorch(a):
    """
    TSVC function s318 - finding maximum absolute value and its index
    
    Original C code:
    for (int nl = 0; nl < iterations/2; nl++) {
        k = 0;
        index = 0;
        max = ABS(a[0]);
        k += inc;
        for (int i = 1; i < LEN_1D; i++) {
            if (ABS(a[k]) <= max) {
                goto L5;
            }
            index = i;
            max = ABS(a[k]);
    L5:
            k += inc;
        }
        chksum = max + (real_t) index;
    }
    
    Arrays used: a (r)
    """
    a = a.contiguous()
    
    # Initialize variables
    index = torch.tensor(0, dtype=torch.int32, device=a.device)
    max_val = torch.abs(a[0])
    
    # Loop through array starting from index 1
    for i in range(1, len(a)):
        abs_val = torch.abs(a[i])
        # If current absolute value is greater than max, update both max and index
        if abs_val > max_val:
            index = torch.tensor(i, dtype=torch.int32, device=a.device)
            max_val = abs_val
    
    # Calculate checksum (though not returned, this maintains equivalence)
    chksum = max_val + index.float()
    
    return a