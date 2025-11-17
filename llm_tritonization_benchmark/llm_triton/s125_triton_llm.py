import torch
import triton
import triton.language as tl

@triton.jit
def s125_kernel(
    aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load data with masking
    aa = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    bb = tl.load(bb_ptr + offsets, mask=mask, other=0.0)
    cc = tl.load(cc_ptr + offsets, mask=mask, other=0.0)
    
    # Compute aa + bb * cc (fused multiply-add for better performance)
    result = aa + bb * cc
    
    # Store result to flattened array
    tl.store(flat_2d_array_ptr + offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    """
    Triton implementation of TSVC s125 - 2D to 1D array flattening with computation.
    Optimized for GPU with coalesced memory access and fused operations.
    """
    # Ensure tensors are contiguous and on GPU
    aa = aa.contiguous()
    bb = bb.contiguous() 
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    # Get total number of elements
    n_elements = aa.numel()
    
    # Choose block size - power of 2 for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened memory layout for optimal access patterns
    s125_kernel[grid](
        aa.view(-1),  # Flatten to 1D for coalesced access
        bb.view(-1),
        cc.view(-1), 
        flat_2d_array,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return flat_2d_array