import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction to find max value and its index
    # Use single thread since it's a sequential reduction
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                current_idx = block_start + i
                current_val = tl.load(a_ptr + current_idx)
                
                if current_val > max_val:
                    max_val = current_val
                    max_idx = current_idx
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Result tensor to store [max_value, max_index]
    result = torch.zeros(2, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block for reduction
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    
    return max_idx, max_val