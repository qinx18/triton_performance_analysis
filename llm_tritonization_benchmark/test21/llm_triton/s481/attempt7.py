import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, valid_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < valid_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Phase 1: Find exit point
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        valid_elements = exit_idx
    else:
        valid_elements = n_elements
    
    # Phase 2: Process valid elements if any
    if valid_elements > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(valid_elements, BLOCK_SIZE),)
        s481_kernel[grid](a, b, c, valid_elements, BLOCK_SIZE=BLOCK_SIZE)