import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(a_ptr, b_ptr, indices_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    b_offsets = block_start + offsets
    mask = b_offsets < n_elements
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    pos_mask = b_vals > 0.0
    
    tl.store(indices_ptr + b_offsets, pos_mask.to(tl.int32), mask=mask)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Create temporary array to mark positive elements
    indices = torch.zeros(n_elements, dtype=torch.int32, device=b.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Mark positive elements
    s341_kernel[grid](a, b, indices, n_elements, BLOCK_SIZE)
    
    # Find positive values and pack them
    pos_mask = indices.bool()
    pos_values = b[pos_mask]
    
    # Clear array a first
    a.fill_(0.0)
    
    # Pack positive values sequentially
    if pos_values.numel() > 0:
        a[:pos_values.numel()] = pos_values
    
    return a