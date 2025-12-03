import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel to find max value and its index
    # We need to process the entire array in one block group
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max value and index
    max_val = tl.load(a_ptr)  # a[0]
    max_idx = 0
    
    # Process array in chunks
    for start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Load current chunk
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find local maximum in this chunk
        for i in range(BLOCK_SIZE):
            if start + i < n_elements:
                current_val = tl.load(a_ptr + start + i)
                if current_val > max_val:
                    max_val = current_val
                    max_idx = start + i
    
    # Only first program stores the result
    if pid == 0:
        chksum = max_val + max_idx
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_idx)
        tl.store(result_ptr + 2, chksum)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor to store [max_val, max_idx, chksum]
    result = torch.zeros(3, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single program to handle the reduction
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0].item()
    max_idx = int(result[1].item())
    chksum = result[2].item()
    
    return max_val, max_idx, chksum