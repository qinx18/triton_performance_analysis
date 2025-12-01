import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(
    a_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to find the last index where a[i] < 0.
    Uses reduction to find maximum index across all blocks.
    """
    # Get block start index
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements
    mask = offsets < n_elements
    
    # Load values with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)  # Use 1.0 as default to avoid < 0 condition
    
    # Find elements < 0 and get their indices
    neg_mask = (a_vals < 0.0) & mask
    
    # Create indices for this block, use -1 for invalid positions
    indices = tl.where(neg_mask, offsets, -1)
    
    # Find maximum valid index in this block
    block_max = tl.max(indices)
    
    # Store result for this block (will be reduced later)
    if tl.program_id(0) == 0:
        tl.store(result_ptr + tl.program_id(0), block_max)
    else:
        tl.store(result_ptr + tl.program_id(0), block_max)

@triton.jit
def s331_reduce_kernel(
    partial_results_ptr,
    final_result_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Reduction kernel to find the global maximum index.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load partial results
    partial_vals = tl.load(partial_results_ptr + offsets, mask=mask, other=-1)
    
    # Find global maximum
    global_max = tl.max(partial_vals)
    
    # Store final result
    if tl.program_id(0) == 0:
        tl.store(final_result_ptr, global_max)

def s331_triton(a):
    """
    Triton implementation of TSVC s331 function.
    Finds the last (maximum) index where a[i] < 0, or -1 if none exist.
    Uses two-stage reduction for efficient parallel computation.
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    if n_elements == 0:
        return a
    
    # Choose block size based on array size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for partial results
    partial_results = torch.full((num_blocks,), -1, dtype=torch.long, device=a.device)
    
    # Launch first kernel to find partial maximums
    grid = (num_blocks,)
    s331_kernel[grid](
        a,
        partial_results,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # If we have multiple blocks, reduce the partial results
    if num_blocks > 1:
        final_result = torch.tensor([-1], dtype=torch.long, device=a.device)
        reduce_block_size = min(1024, triton.next_power_of_2(num_blocks))
        
        s331_reduce_kernel[(1,)](
            partial_results,
            final_result,
            num_blocks,
            BLOCK_SIZE=reduce_block_size,
        )
    else:
        final_result = partial_results[:1]
    
    # The actual computation is complete, just return the input tensor
    # (matching the baseline behavior of returning the input)
    return a