import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find maximum value and its index using reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    if tl.program_id(0) == 0:
        max_val = tl.load(a_ptr)
        max_idx = 0
        
        # Process array in blocks
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load block of data
            block_data = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
            
            # Find max in current block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    val = tl.load(a_ptr + block_start + i)
                    if val > max_val:
                        max_val = val
                        max_idx = block_start + i
        
        # Store results
        tl.store(result_ptr, max_val)
        tl.store(result_ptr + 1, max_idx.to(tl.float32))

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor [max_value, max_index]
    result = torch.zeros(2, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s315_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    max_val = result[0]
    max_idx = result[1]
    chksum = max_val + max_idx
    
    return chksum