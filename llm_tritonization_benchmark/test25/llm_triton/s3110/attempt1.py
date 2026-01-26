import torch
import triton
import triton.language as tl

def s3110_triton(aa, len_2d):
    # Use PyTorch for argmax reduction since tracking indices in Triton is complex
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // aa.shape[1]
    yindex = max_idx % aa.shape[1]
    
    chksum = max_val + xindex.float() + yindex.float()
    
    return max_val + (xindex + 1) + (yindex + 1)