import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential reduction to find max value and its index
    # Use single thread to maintain correctness
    pid = tl.program_id(0)
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
        
        # Load block of values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=-float('inf'))
        
        # Check each element in the block
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            current_val = tl.load(a_ptr + block_start + i)
            if current_val > max_val:
                max_val = current_val
                max_idx = block_start + i
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor to store max value and index
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since this is a sequential reduction
    grid = (1,)
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0]
    max_idx = result[1].int()
    chksum = max_val + max_idx.float()
    
    return max_val, max_idx, chksum