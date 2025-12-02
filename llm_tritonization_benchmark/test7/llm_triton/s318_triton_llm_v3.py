import torch
import triton
import triton.language as tl

@triton.jit
def s318_kernel(a_ptr, max_val_ptr, max_idx_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum absolute value and its index
    # Each block processes the entire array sequentially to maintain correctness
    block_id = tl.program_id(0)
    
    if block_id > 0:
        return
    
    # Initialize with first element
    k = 0
    current_max = tl.abs(tl.load(a_ptr + k))
    current_idx = 0
    k += inc
    
    # Process remaining elements
    for i in range(1, n_elements):
        if k < n_elements * inc:
            abs_val = tl.abs(tl.load(a_ptr + k))
            # Update max and index if we found a larger value
            if abs_val > current_max:
                current_max = abs_val
                current_idx = i
        k += inc
    
    # Store results
    tl.store(max_val_ptr, current_max)
    tl.store(max_idx_ptr, current_idx)

def s318_triton(a, inc):
    n_elements = a.shape[0]
    
    # Create output tensors
    max_val = torch.zeros(1, dtype=a.dtype, device=a.device)
    max_idx = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 128
    grid = (1,)  # Single block to maintain sequential correctness
    
    s318_kernel[grid](
        a, max_val, max_idx,
        inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return max_val.item() + max_idx.item()