import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Each block processes BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n
    
    # Load values with mask
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find local maximum and its index
    local_max = tl.max(abs_vals)
    local_max_mask = abs_vals == local_max
    
    # Find the first occurrence of max in this block
    local_indices = tl.where(local_max_mask & mask, offsets, n)
    local_min_idx = tl.min(local_indices)
    
    # Store results for this block
    block_id = tl.program_id(0)
    tl.store(result_ptr + block_id * 2, local_max)
    tl.store(result_ptr + block_id * 2 + 1, local_min_idx)

def s318_triton(a, inc):
    n = a.shape[0]
    
    # Handle stride access by creating strided view
    if inc == 1:
        strided_a = a
    else:
        # Create indices for strided access
        max_k = ((n - 1) // inc + 1) * inc
        if max_k >= n:
            max_k = n - 1
        num_elements = (max_k // inc) + 1
        indices = torch.arange(0, num_elements, device=a.device) * inc
        indices = indices[indices < n]
        if len(indices) == 0:
            return torch.tensor(1.0, device=a.device)
        strided_a = a[indices]
    
    strided_n = strided_a.shape[0]
    if strided_n == 0:
        return torch.tensor(1.0, device=a.device)
    
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(strided_n, BLOCK_SIZE)
    
    # Result buffer: [max_val, index] for each block
    result = torch.zeros(num_blocks * 2, device=a.device, dtype=torch.float32)
    
    grid = (num_blocks,)
    s318_kernel[grid](strided_a, result, inc, strided_n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce across blocks
    max_vals = result[::2]  # Even indices are max values
    indices = result[1::2].long()  # Odd indices are indices
    
    # Find global maximum
    global_max = torch.max(max_vals)
    global_max_block = torch.argmax(max_vals)
    global_max_idx = indices[global_max_block]
    
    # Convert back to original array indexing if needed
    if inc != 1:
        # The index in strided array needs to be converted to loop iteration
        # In C code: for i from 1, so we need the iteration number
        original_index = global_max_idx.item()
    else:
        original_index = global_max_idx.item()
    
    # The C code returns max + index + 1
    # But index in C code is the loop variable i, which starts from 1
    # So we need to adjust
    return global_max + original_index + 1