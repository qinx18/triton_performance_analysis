import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(
    aa_ptr, bb_ptr, cc_ptr, output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1115: aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
    Uses 1D indexing for efficient memory access patterns
    """
    # Get program ID and compute linear index
    pid = tl.program_id(axis=0)
    
    # Compute the range of elements this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < M * N
    
    # Load data with masking
    aa_vals = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + offsets, mask=mask, other=0.0)
    
    # Convert linear index to 2D coordinates
    i = offsets // N
    j = offsets % N
    
    # Compute transposed index: cc[j][i] requires accessing cc at position j*M + i
    cc_transposed_idx = j * M + i
    cc_mask = mask & (cc_transposed_idx < M * N)
    cc_vals = tl.load(cc_ptr + cc_transposed_idx, mask=cc_mask, other=0.0)
    
    # Compute: aa[i][j] * cc[j][i] + bb[i][j]
    result = aa_vals * cc_vals + bb_vals
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def s1115_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s1115
    Optimized for GPU with coalesced memory access and efficient transpose handling
    """
    # Ensure tensors are contiguous and get dimensions
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    
    M, N = aa.shape
    total_elements = M * N
    
    # Create output tensor
    output = torch.empty_like(aa)
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1115_kernel[grid](
        aa, bb, cc, output,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output