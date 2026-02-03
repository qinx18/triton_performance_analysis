import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block processes BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Check which elements are < 0
    neg_mask = vals < 0.0
    combined_mask = mask & neg_mask
    
    # Find the maximum index in this block that satisfies the condition
    max_idx = -1
    indices = current_offsets
    
    # Use tl.where to get valid indices or -1
    valid_indices = tl.where(combined_mask, indices, -1)
    
    # Find maximum valid index in this block
    block_max = tl.max(valid_indices)
    
    # Atomically update global maximum if we found a valid index
    if block_max >= 0:
        tl.atomic_max(result_ptr, block_max)

def s331_triton(a):
    N = a.shape[0]
    
    # Output tensor for result, initialized to -1
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    s331_kernel[(num_blocks,)](
        a,
        result,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()