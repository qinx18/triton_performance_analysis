import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, max_val_ptr, max_idx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize reduction values
    block_max = float('-inf')
    block_idx = 0
    
    # Process elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                val = tl.load(a_ptr + block_start + i)
                if val > block_max:
                    block_max = val
                    block_idx = block_start + i

    # Store results (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(max_val_ptr, block_max)
        tl.store(max_idx_ptr, block_idx)

def s315_triton(a):
    # Use PyTorch for argmax reduction as recommended
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the exact value as specified in C code: index + x + 1
    return float(max_idx) + float(max_val) + 1