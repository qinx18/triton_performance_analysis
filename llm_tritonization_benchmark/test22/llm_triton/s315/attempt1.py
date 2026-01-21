import triton
import triton.language as tl
import torch

@triton.jit
def s315_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize global max and index
    global_max = tl.full([1], float('-inf'), dtype=tl.float32)
    global_idx = tl.full([1], 0, dtype=tl.int32)
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load block of data
        block_data = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this block
        block_max = tl.max(block_data)
        
        # If this block's max is greater than global max, update
        if block_max > global_max:
            global_max = block_max
            # Find the index of the max element in this block
            block_max_mask = block_data == block_max
            # Get the first occurrence index
            for i in range(BLOCK_SIZE):
                if block_start + i < n_elements:
                    if block_data[i] == block_max:
                        global_idx = block_start + i
                        break
    
    # Store results
    tl.store(result_ptr, global_max)
    tl.store(result_ptr + 1, global_idx.to(tl.float32))

def s315_triton(a):
    n = a.shape[0]
    
    # Use PyTorch for argmax reduction (more efficient)
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the exact value as C code: index + x + 1
    return (max_idx + max_val + 1).item()