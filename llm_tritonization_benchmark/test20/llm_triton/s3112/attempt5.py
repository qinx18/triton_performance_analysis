import triton
import triton.language as tl
import torch

@triton.jit
def s3112_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load previous cumulative sum from the last element of previous block
    prev_sum = 0.0
    if pid > 0:
        prev_idx = block_start - 1
        prev_sum = tl.load(b_ptr + prev_idx)
    
    # Load current block data
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute cumulative sum within block using scan pattern
    results = tl.zeros_like(vals)
    
    # Initialize first element
    first_mask = offsets == 0
    first_val = tl.sum(tl.where(first_mask & mask, vals, 0.0))
    first_result = prev_sum + first_val
    results = tl.where(first_mask, first_result, results)
    
    # Process remaining elements
    for step in range(1, BLOCK_SIZE):
        curr_mask = offsets == step
        if tl.sum((curr_mask & mask).to(tl.int32)) > 0:
            prev_mask = offsets == (step - 1)
            prev_result = tl.sum(tl.where(prev_mask, results, 0.0))
            curr_val = tl.sum(tl.where(curr_mask & mask, vals, 0.0))
            curr_result = prev_result + curr_val
            results = tl.where(curr_mask, curr_result, results)
    
    tl.store(b_ptr + indices, results, mask=mask)

def s3112_triton(a, b):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernels sequentially to maintain dependencies
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s3112_kernel[grid](a, b, n, BLOCK_SIZE)
    
    return b[n-1].item()