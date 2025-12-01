import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction operation to find max value and its index
    # We need to use a single block to handle the reduction properly
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize with first element
    max_val = tl.load(a_ptr)
    max_idx = 0
    
    # Process elements in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load block of data
        vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
        
        # Find local max in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                current_val = tl.load(a_ptr + block_start + i)
                if current_val > max_val:
                    max_val = current_val
                    max_idx = block_start + i
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Result tensor to store [max_value, max_index]
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block for reduction
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0]
    max_idx = int(result[1])
    chksum = max_val + max_idx
    
    return max_val, max_idx, chksum