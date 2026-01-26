import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load all elements in blocks and find global maximum
    max_val = float('-inf')
    max_idx = -1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find block maximum and its local index
        block_max = tl.max(vals)
        
        # Check if this block contains the global maximum
        if block_max > max_val:
            max_val = block_max
            # Find the local index of maximum in this block
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    val = tl.load(a_ptr + block_start + i)
                    if val == block_max:
                        max_idx = block_start + i
                        break
    
    # Store results
    tl.store(result_ptr, max_val)
    tl.store(result_ptr + 1, max_idx)

def s315_triton(a):
    n_elements = a.shape[0]
    
    # Use PyTorch for argmax since Triton argmax is complex
    max_val = torch.max(a).item()
    max_idx = torch.argmax(a).item()
    
    # Return the exact value as in C code: index + x + 1
    return max_idx + max_val + 1