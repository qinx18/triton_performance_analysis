import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to use a reduction approach since this is inherently sequential
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check condition and create valid indices
    condition = a_vals < 0.0
    valid_indices = tl.where(condition & mask, offsets, -1)
    
    # Find the maximum valid index in this block
    max_idx = tl.max(valid_indices)
    
    # Store the result for this block
    tl.store(result_ptr + pid, max_idx)

def s331_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block results
    block_results = torch.full((grid_size,), -1, dtype=torch.int32, device=a.device)
    
    # Launch kernel
    s331_kernel[(grid_size,)](
        a, block_results, n_elements, BLOCK_SIZE
    )
    
    # Find the maximum index across all blocks
    j = torch.max(block_results).item()
    
    # Convert to float for chksum (matching C code behavior)
    chksum = float(j)
    
    return j