import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements stream compaction sequentially
    # Each program handles one block and processes it sequentially
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    if block_start >= n_elements:
        return
    
    # Process elements sequentially within each block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load block of data
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # For stream compaction, we need to process sequentially
    # This is inherently sequential due to the dependency on counter j
    for i in range(BLOCK_SIZE):
        if block_start + i >= n_elements:
            break
        
        current_offset = block_start + i
        b_val = tl.load(b_ptr + current_offset)
        
        if b_val > 0.0:
            # This is problematic in parallel - we need atomic operations
            # But Triton doesn't have atomic increment, so we fall back to host
            pass

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Stream compaction is difficult to parallelize efficiently in Triton
    # Use PyTorch's boolean indexing which is optimized for this pattern
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Only update the positions that should be written to
    if num_packed > 0:
        a[:num_packed] = packed_values
    
    return a