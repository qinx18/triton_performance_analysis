import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to use a reduction-like approach since this is inherently sequential
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check condition and create valid indices
    condition = a_vals < 0.0
    valid_indices = tl.where(condition & mask, offsets, -1)
    
    # Find the maximum valid index in this block
    block_max_idx = tl.max(valid_indices)
    
    # Store the block result
    tl.store(output_ptr + pid, block_max_idx)

@triton.jit  
def s331_final_reduction_kernel(partial_results_ptr, output_ptr, num_blocks):
    # Find the maximum across all blocks
    max_idx = -1
    for i in range(num_blocks):
        block_result = tl.load(partial_results_ptr + i)
        max_idx = tl.maximum(max_idx, block_result)
    
    tl.store(output_ptr, max_idx)

def s331_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Partial results from each block
    partial_results = torch.empty(num_blocks, dtype=torch.int32, device=a.device)
    
    # Launch first kernel to get partial results
    s331_kernel[(num_blocks,)](
        a, partial_results, n_elements, BLOCK_SIZE
    )
    
    # Final result
    result = torch.empty(1, dtype=torch.int32, device=a.device)
    
    # Launch reduction kernel
    s331_final_reduction_kernel[(1,)](
        partial_results, result, num_blocks
    )
    
    j = result.item()
    chksum = float(j)
    
    return j