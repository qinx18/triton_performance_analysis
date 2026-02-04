import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel_pass1(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute prefix sum within block
    prefix_sums = tl.cumsum(vals, axis=0)
    
    # Store results
    tl.store(b_ptr + current_offsets, prefix_sums, mask=mask)

@triton.jit
def s3112_kernel_pass2(b_ptr, block_offsets_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    if block_id == 0:
        return
    
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load block offset
    block_offset = tl.load(block_offsets_ptr + block_id - 1)
    
    # Load current values and add offset
    current_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    updated_vals = current_vals + block_offset
    
    # Store updated values
    tl.store(b_ptr + current_offsets, updated_vals, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Pass 1: Compute local prefix sums
    grid = (num_blocks,)
    s3112_kernel_pass1[grid](a, b, n, BLOCK_SIZE)
    
    if num_blocks > 1:
        # Extract block totals
        block_ends = torch.arange(BLOCK_SIZE - 1, n, BLOCK_SIZE, device=b.device)
        block_ends = block_ends.clamp(max=n-1)
        block_totals = b[block_ends]
        
        # Compute cumulative offsets
        block_offsets = torch.cumsum(block_totals, dim=0)
        
        # Pass 2: Add block offsets
        s3112_kernel_pass2[grid](b, block_offsets, n, BLOCK_SIZE)
    
    # Return final sum (last element)
    return b[n-1].item()