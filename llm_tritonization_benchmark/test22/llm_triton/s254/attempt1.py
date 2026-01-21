import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Expand scalar x using sequential processing"""
    pid = tl.program_id(0)
    
    # Only use first thread to compute expansion sequentially
    if pid == 0:
        x_val = tl.load(b_ptr + (N - 1))  # x = b[LEN_1D-1]
        
        # Process in blocks sequentially
        for block_start in range(0, N, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            # Store current x_val for all positions in this block
            tl.store(x_expanded_ptr + current_offsets, x_val, mask=mask)
            
            # Update x_val to b[i] for the last valid position in this block
            if block_start + BLOCK_SIZE - 1 < N:
                # Full block
                x_val = tl.load(b_ptr + (block_start + BLOCK_SIZE - 1))
            elif mask.any():
                # Partial block - get last valid position
                last_valid = min(block_start + BLOCK_SIZE - 1, N - 1)
                x_val = tl.load(b_ptr + last_valid)

@triton.jit
def s254_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Compute a[i] = (b[i] + x_expanded[i]) * 0.5 in parallel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < N
    
    # Load b values and expanded x values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = (b[i] + x) * 0.5
    result = (b_vals + x_vals) * 0.5
    
    # Store results
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s254_triton(a, b):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded x
    x_expanded = torch.zeros_like(b)
    
    # Phase 1: Expand scalar x
    grid = (1,)  # Single thread block
    s254_expand_x_kernel[grid](b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute results in parallel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s254_compute_kernel[grid](a, b, x_expanded, N, BLOCK_SIZE=BLOCK_SIZE)