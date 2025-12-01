import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(
    aa_ptr,
    max_val_ptr,
    max_idx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for finding maximum element and its index.
    Uses block-level reduction to efficiently find global maximum.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking for out-of-bounds elements
    # Use -inf for masked elements so they don't affect the max
    vals = tl.load(aa_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find local maximum and its index within the block
    local_max = tl.max(vals, axis=0)
    
    # Find the index of the maximum value within this block
    is_max = vals == local_max
    local_indices = tl.arange(0, BLOCK_SIZE)
    # Get the first occurrence of the maximum value
    max_mask = is_max & mask
    local_max_idx = tl.min(tl.where(max_mask, local_indices, n_elements))
    global_max_idx = block_start + local_max_idx
    
    # Store results (will be reduced across blocks later)
    tl.store(max_val_ptr + pid, local_max)
    tl.store(max_idx_ptr + pid, global_max_idx)

@triton.jit
def reduce_kernel(
    max_vals_ptr,
    max_indices_ptr,
    final_max_ptr,
    final_idx_ptr,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel to reduce results from multiple blocks to find global maximum.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_blocks
    
    # Load block results
    block_maxes = tl.load(max_vals_ptr + offsets, mask=mask, other=float('-inf'))
    block_indices = tl.load(max_indices_ptr + offsets, mask=mask, other=0)
    
    # Find global maximum
    global_max = tl.max(block_maxes, axis=0)
    
    # Find which block had the global maximum
    is_global_max = block_maxes == global_max
    final_index = tl.min(tl.where(is_global_max & mask, block_indices, 2**31-1))
    
    # Store final results
    tl.store(final_max_ptr, global_max)
    tl.store(final_idx_ptr, final_index)

def s3110_triton(aa):
    """
    Triton implementation of TSVC s3110 - finding maximum element and its indices.
    Uses two-stage reduction: block-level then global reduction for efficiency.
    """
    aa = aa.contiguous()
    
    # Flatten the array for processing
    flat_aa = aa.view(-1)
    n_elements = flat_aa.numel()
    
    # Choose block size and calculate number of blocks
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for block results
    block_max_vals = torch.empty(n_blocks, dtype=aa.dtype, device=aa.device)
    block_max_indices = torch.empty(n_blocks, dtype=torch.int32, device=aa.device)
    
    # Launch first kernel to find maximum in each block
    grid = (n_blocks,)
    s3110_kernel[grid](
        flat_aa,
        block_max_vals,
        block_max_indices,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Allocate final result storage
    final_max = torch.empty(1, dtype=aa.dtype, device=aa.device)
    final_idx = torch.empty(1, dtype=torch.int32, device=aa.device)
    
    # Launch reduction kernel to find global maximum
    reduce_block_size = min(1024, triton.next_power_of_2(n_blocks))
    reduce_kernel[(1,)](
        block_max_vals,
        block_max_indices,
        final_max,
        final_idx,
        n_blocks,
        BLOCK_SIZE=reduce_block_size,
    )
    
    # Convert results back to match PyTorch implementation
    max_val = final_max.item()
    max_idx = final_idx.item()
    
    # Convert linear index to 2D coordinates (matching baseline behavior)
    LEN_2D = aa.size(0)
    xindex = max_idx // LEN_2D
    yindex = max_idx % LEN_2D
    
    # Calculate checksum (matching baseline)
    chksum = max_val + float(xindex) + float(yindex)
    
    return aa