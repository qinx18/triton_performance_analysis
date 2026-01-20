import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # Each block processes a chunk of the array
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Load values with masking
    mask = indices < n
    k_indices = indices * inc
    valid_k = k_indices < n
    combined_mask = mask & valid_k
    
    # Load absolute values
    values = tl.load(a_ptr + k_indices, mask=combined_mask, other=0.0)
    abs_values = tl.abs(values)
    
    # Set invalid values to negative infinity so they don't affect max
    abs_values = tl.where(combined_mask, abs_values, float('-inf'))
    
    # Find local maximum and its index
    local_max = tl.max(abs_values)
    local_argmax = tl.argmax(abs_values, axis=0)
    
    # Convert local argmax to global index
    global_argmax = block_start + local_argmax
    
    # Store results in shared memory for reduction across blocks
    max_ptr = tl.static_alloc(1, dtype=tl.float32)
    idx_ptr = tl.static_alloc(1, dtype=tl.int32)
    
    tl.store(max_ptr, local_max)
    tl.store(idx_ptr, global_argmax)

def s318_triton(a, inc):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # For this reduction, we need to handle it differently since we need both max and argmax
    # Use PyTorch for the strided access and reduction
    k_indices = torch.arange(0, n, dtype=torch.int64, device=a.device) * inc
    valid_mask = k_indices < n
    
    if not valid_mask.any():
        return torch.tensor(1.0, device=a.device)
    
    # Get valid indices and corresponding values
    valid_k = k_indices[valid_mask]
    valid_a = a[valid_k]
    abs_valid_a = torch.abs(valid_a)
    
    # Find maximum absolute value and its index
    max_val = torch.max(abs_valid_a)
    max_idx = torch.argmax(abs_valid_a)
    
    # The C code uses i as the loop index (starting from 1), not the array index
    # Since we start from i=1 in the C loop, we need to add 1 to the argmax result
    result_idx = max_idx + 1
    
    # Return max + index + 1 as specified in the C code
    return max_val + result_idx + 1