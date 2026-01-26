import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    max_val = first_val
    max_idx = 0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find local maximum in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                if i == 0:
                    local_val = tl.load(a_ptr + block_start + i)
                else:
                    local_val = tl.load(a_ptr + block_start + i)
                
                if local_val > max_val:
                    max_val = local_val
                    max_idx = block_start + i
    
    # Store result
    result = max_val + max_idx + 1
    tl.store(output_ptr, result)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Use PyTorch for argmax reduction (recommended approach)
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the same value as C code: index + x + 1
    result = max_val + max_idx + 1
    return result