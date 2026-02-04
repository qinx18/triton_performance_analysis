import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel(a_ptr, b_ptr, block_offsets_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_id * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute prefix sum within the block
    prefix_sums = tl.cumsum(vals, axis=0)
    
    # Add offset from previous blocks if not the first block
    if block_id > 0:
        block_offset = tl.load(block_offsets_ptr + block_id - 1)
        prefix_sums = prefix_sums + block_offset
    
    # Store the prefix sums to output
    tl.store(b_ptr + current_offsets, prefix_sums, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n))
    n_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # First pass: compute local prefix sums
    grid = (n_blocks,)
    block_offsets = torch.zeros(n_blocks, device=a.device, dtype=a.dtype)
    
    s3112_kernel[grid](a, b, block_offsets, n, BLOCK_SIZE)
    
    # If multiple blocks, compute block offsets and apply second pass
    if n_blocks > 1:
        # Extract block totals (last element of each block)
        block_ends = torch.arange(BLOCK_SIZE - 1, n, BLOCK_SIZE, device=a.device)
        block_ends = block_ends.clamp(max=n-1)
        block_totals = b[block_ends]
        
        # Compute cumulative block offsets
        if len(block_totals) > 1:
            block_offsets[1:] = torch.cumsum(block_totals[:-1], dim=0)
            
            # Second pass: add block offsets
            s3112_kernel[grid](a, b, block_offsets, n, BLOCK_SIZE)
    
    # Return the final sum (last element of b)
    return b[n-1].item()