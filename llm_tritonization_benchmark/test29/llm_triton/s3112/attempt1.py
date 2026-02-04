import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel_local(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Compute local prefix sums for each block"""
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute local prefix sum
    prefix_sums = tl.cumsum(vals, axis=0)
    
    # Store results
    tl.store(b_ptr + current_offsets, prefix_sums, mask=mask)

@triton.jit
def s3112_kernel_global(b_ptr, offsets_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Add global offsets to each block"""
    block_id = tl.program_id(0)
    if block_id == 0:
        return
    
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load current values
    vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load global offset for this block
    global_offset = tl.load(offsets_ptr + block_id - 1)
    
    # Add global offset
    corrected_vals = vals + global_offset
    
    # Store results
    tl.store(b_ptr + current_offsets, corrected_vals, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # First pass: compute local prefix sums
    grid = (num_blocks,)
    s3112_kernel_local[grid](a, b, n, BLOCK_SIZE)
    
    if num_blocks > 1:
        # Extract block totals using tensor indexing
        block_ends = torch.arange(BLOCK_SIZE - 1, n, BLOCK_SIZE, device=b.device)
        block_ends = block_ends.clamp(max=n-1)
        block_totals = b[block_ends]
        
        # Compute cumulative offsets
        block_offsets = torch.cumsum(block_totals[:-1], dim=0)
        
        # Second pass: add global offsets
        s3112_kernel_global[grid](b, block_offsets, n, BLOCK_SIZE)
    
    # Return final sum (last element of cumulative sum)
    return b[n-1].item()