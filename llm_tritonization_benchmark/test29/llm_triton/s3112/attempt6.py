import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel_pass1(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = pid * BLOCK_SIZE + offsets
    mask = block_offsets < n_elements
    
    # Load block of data
    vals = tl.load(a_ptr + block_offsets, mask=mask, other=0.0)
    
    # Compute local prefix sum within block
    prefix_sums = tl.cumsum(vals, axis=0)
    
    # Store local prefix sums
    tl.store(b_ptr + block_offsets, prefix_sums, mask=mask)

@triton.jit
def s3112_kernel_pass2(b_ptr, block_offsets_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Skip first block (no offset needed)
    if pid == 0:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = pid * BLOCK_SIZE + offsets
    mask = block_offsets < n_elements
    
    # Load the offset for this block
    offset_val = tl.load(block_offsets_ptr + pid - 1)
    
    # Load current values and add offset
    vals = tl.load(b_ptr + block_offsets, mask=mask)
    vals = vals + offset_val
    
    # Store updated values
    tl.store(b_ptr + block_offsets, vals, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # First pass: compute local prefix sums
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s3112_kernel_pass1[grid](a, b, n, BLOCK_SIZE)
    
    # If only one block, we're done
    if n <= BLOCK_SIZE:
        return b[-1].item()
    
    # Extract block totals (last element of each block)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    block_end_indices = torch.arange(BLOCK_SIZE - 1, n, BLOCK_SIZE, device=b.device)
    block_end_indices = torch.clamp(block_end_indices, max=n-1)
    
    # Get totals and compute cumulative offsets
    block_totals = b[block_end_indices]
    block_offsets = torch.cumsum(block_totals[:-1], dim=0)
    
    # Second pass: add offsets to subsequent blocks
    grid = (num_blocks,)
    s3112_kernel_pass2[grid](b, block_offsets, n, BLOCK_SIZE)
    
    return b[-1].item()