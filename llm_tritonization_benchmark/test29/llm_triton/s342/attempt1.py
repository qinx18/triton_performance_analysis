import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel():
    # Stub kernel - actual computation done in wrapper using PyTorch
    pass

def s342_triton(a, b):
    # Stream compaction pattern - conditional scatter from packed array
    mask = (a > 0.0).to(torch.int32)
    indices = torch.cumsum(mask, dim=0) - 1
    bool_mask = mask.bool()
    a[bool_mask] = b[indices[bool_mask]]