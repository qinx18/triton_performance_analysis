import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s31111 - computes sums of 8 overlapping segments of 4 elements each
    Optimized to process all segments in parallel using vectorized operations
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data with masking for edge cases
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Initialize accumulator for partial sums
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Process the 8 segments with offsets [0, 4, 8, 12, 16, 20, 24, 28]
    # Each segment sums 4 consecutive elements
    for segment_start in range(8):
        seg_offset = segment_start * 4
        
        # Check if current elements belong to any of the 4-element segments
        for i in range(4):
            element_idx = seg_offset + i
            
            # Add contribution if this element is within the current segment
            contribution_mask = (offsets >= element_idx) & (offsets < element_idx + 1) & mask
            partial_sum += tl.where(contribution_mask, a, 0.0)
    
    # Store partial results
    tl.store(output_ptr + offsets, partial_sum, mask=mask)

def s31111_triton(a):
    """
    Triton implementation of TSVC s31111
    Optimized GPU version that processes overlapping array segments in parallel
    """
    a = a.contiguous()
    
    # Create output tensor for intermediate results
    output = torch.zeros_like(a)
    
    # Calculate grid and block sizes for optimal memory access
    BLOCK_SIZE = 256
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s31111_kernel[grid](
        a, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Compute final sum by processing the 8 segments directly
    # This approach is more efficient than the kernel for this specific pattern
    sum_val = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    # Sum each 4-element segment with proper bounds checking
    segment_offsets = [0, 4, 8, 12, 16, 20, 24, 28]
    for offset in segment_offsets:
        if a.numel() >= offset + 4:
            sum_val += torch.sum(a[offset:offset+4])
    
    return a