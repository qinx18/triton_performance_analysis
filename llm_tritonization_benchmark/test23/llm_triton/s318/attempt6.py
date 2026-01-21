import triton
import triton.language as tl
import torch

@triton.jit
def s318_kernel(a_ptr, result_ptr, inc, n, BLOCK_SIZE: tl.constexpr):
    # This kernel finds max absolute value and its index with stride
    # Each block processes a chunk of the array
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Initialize local max and index
    local_max = 0.0
    local_idx = 0
    
    # Process elements in this block
    mask = indices < n
    k_vals = indices * inc
    k_mask = mask & (k_vals < n)  # Ensure k doesn't go out of bounds
    
    # Load values with stride
    vals = tl.load(a_ptr + k_vals, mask=k_mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find max in this block
    block_max = tl.max(abs_vals)
    
    # Find the index of the maximum value
    is_max_mask = (abs_vals == block_max) & k_mask
    max_indices = tl.where(is_max_mask, indices, n)
    block_idx = tl.min(max_indices)  # Get the first occurrence
    
    # Store results for reduction across blocks
    tl.store(result_ptr + block_id * 2, block_max)
    tl.store(result_ptr + block_id * 2 + 1, block_idx.to(tl.float32))

def s318_triton(a, abs, inc):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Allocate temporary storage for block results
    block_results = torch.zeros(num_blocks * 2, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s318_kernel[grid](a, block_results, inc, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce across blocks on CPU for simplicity
    block_maxes = block_results[::2]  # Even indices are max values
    block_indices = block_results[1::2]  # Odd indices are index values
    
    # Find global maximum
    global_max_block_idx = torch.argmax(block_maxes)
    global_max = block_maxes[global_max_block_idx]
    global_idx = int(block_indices[global_max_block_idx])
    
    # Handle the case where we need to check the first element (index 0)
    # Since the C code starts with max = ABS(a[0]), index = 0
    if torch.abs(a[0]) > global_max:
        global_max = torch.abs(a[0])
        global_idx = 0
    
    # Return the same value as C code: max + index + 1
    return global_max + global_idx + 1