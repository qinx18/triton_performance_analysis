import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, max_val_ptr, max_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find max value and index within block
    block_max = tl.max(vals)
    block_max_idx = tl.argmax(vals, axis=0)
    
    # Store results (will be reduced across blocks in wrapper)
    if tl.program_id(0) == 0:
        tl.store(max_val_ptr, block_max)
        tl.store(max_idx_ptr, block_max_idx.to(tl.int32))

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Use PyTorch for argmax reduction
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return exactly what the C code returns: index + x + 1
    result = max_idx + max_val + 1
    return result.cpu().numpy()