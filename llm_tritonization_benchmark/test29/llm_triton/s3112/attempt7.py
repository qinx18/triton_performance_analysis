import torch
import triton
import triton.language as tl

@triton.jit
def s3112_kernel_pass1(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = pid * BLOCK_SIZE + offsets
    mask = block_offsets < n_elements
    
    vals = tl.load(a_ptr + block_offsets, mask=mask, other=0.0)
    prefix_sums = tl.cumsum(vals, axis=0)
    tl.store(b_ptr + block_offsets, prefix_sums, mask=mask)

@triton.jit
def s3112_kernel_pass2(b_ptr, block_offsets_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = pid * BLOCK_SIZE + offsets
    mask = block_offsets < n_elements
    
    offset_val = tl.load(block_offsets_ptr + pid - 1)
    vals = tl.load(b_ptr + block_offsets, mask=mask)
    vals = vals + offset_val
    tl.store(b_ptr + block_offsets, vals, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s3112_kernel_pass1[grid](a, b, n, BLOCK_SIZE)
    
    if n <= BLOCK_SIZE:
        return b[-1].item()
    
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Get last valid element from each block
    block_last_indices = []
    for i in range(num_blocks):
        last_idx_in_block = min((i + 1) * BLOCK_SIZE - 1, n - 1)
        block_last_indices.append(last_idx_in_block)
    
    block_last_indices = torch.tensor(block_last_indices, device=b.device, dtype=torch.long)
    block_totals = b[block_last_indices]
    
    # Compute cumulative sum of all but the last block total
    if num_blocks > 1:
        block_offsets = torch.cumsum(block_totals[:-1], dim=0)
        
        grid = (num_blocks,)
        s3112_kernel_pass2[grid](b, block_offsets, n, BLOCK_SIZE)
    
    return b[-1].item()