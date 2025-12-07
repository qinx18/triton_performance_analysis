import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max value and its index
    # We need to use a single thread block to maintain correctness
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in current block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                current_val = tl.load(a_ptr + block_start + i)
                if current_val > max_val:
                    max_val = current_val
                    max_idx = block_start + i
    
    # Store result as max_val + max_idx
    chksum = max_val + max_idx
    tl.store(result_ptr, chksum)

def s315_triton(a):
    n_elements = a.shape[0]
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single block for reduction
    
    s315_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result