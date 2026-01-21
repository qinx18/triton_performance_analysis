import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with negative value so any real abs value will be larger
    block_max = tl.full([BLOCK_SIZE], -1.0, tl.float32)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Update block_max where mask is true
        block_max = tl.where(mask, tl.maximum(block_max, abs_vals), block_max)
    
    # Reduce within block
    final_max = tl.max(block_max)
    
    # Store result (first thread stores)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, final_max)

@triton.jit
def s3113_reduction_kernel(partial_results_ptr, output_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    vals = tl.load(partial_results_ptr + offsets, mask=mask, other=-1.0)
    final_max = tl.max(vals)
    
    if tl.program_id(0) == 0:
        tl.store(output_ptr, final_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    if num_blocks == 1:
        # Single block case
        output = torch.zeros(1, dtype=a.dtype, device=a.device)
        grid = (1,)
        s3113_kernel[grid](a, output, n, BLOCK_SIZE=BLOCK_SIZE)
        return output.item()
    else:
        # Multi-block case - need reduction
        partial_results = torch.full((num_blocks,), -1.0, dtype=a.dtype, device=a.device)
        
        # First pass: compute partial maxima
        @triton.jit
        def s3113_partial_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
            block_id = tl.program_id(0)
            offsets = tl.arange(0, BLOCK_SIZE)
            current_offsets = block_id * BLOCK_SIZE + offsets
            mask = current_offsets < n
            
            vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            abs_vals = tl.abs(vals)
            
            # Set invalid entries to -1 so they don't affect max
            masked_abs_vals = tl.where(mask, abs_vals, -1.0)
            block_max = tl.max(masked_abs_vals)
            
            tl.store(output_ptr + block_id, block_max)
        
        grid = (num_blocks,)
        s3113_partial_kernel[grid](a, partial_results, n, BLOCK_SIZE=BLOCK_SIZE)
        
        # Second pass: reduce partial results
        final_output = torch.zeros(1, dtype=a.dtype, device=a.device)
        grid = (1,)
        s3113_reduction_kernel[grid](partial_results, final_output, num_blocks, BLOCK_SIZE=256)
        
        return final_output.item()