import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, mask_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask_indices = block_start + offsets
    
    # Load elements and compute condition
    mask_load = mask_indices < n_elements
    b_vals = tl.load(b_ptr + mask_indices, mask=mask_load, other=0.0)
    condition = b_vals > 0.0
    
    # Store condition mask
    tl.store(mask_ptr + mask_indices, condition.to(tl.int32), mask=mask_load)

def s341_triton(a, b):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    # Create mask tensor to store condition results
    mask = torch.zeros(n_elements, dtype=torch.int32, device=b.device)
    
    # Launch kernel to compute condition mask
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s341_kernel[grid](b, mask, n_elements, BLOCK_SIZE)
    
    # Convert to boolean and use PyTorch's boolean indexing for stream compaction
    bool_mask = mask.bool()
    packed_values = b[bool_mask]
    num_packed = packed_values.numel()
    
    # Pack positive values into beginning of array a
    if num_packed > 0:
        a[:num_packed] = packed_values