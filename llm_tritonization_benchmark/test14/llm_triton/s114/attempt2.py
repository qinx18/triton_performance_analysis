import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Process valid indices
    valid_indices = tl.where(mask, indices, 0)
    
    # Convert linear indices to (i, j) coordinates
    # For j < i, we map sequentially: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
    
    # Find i using quadratic formula approximation
    i_approx = tl.sqrt(2.0 * valid_indices.to(tl.float32) + 0.25) + 0.5
    i = i_approx.to(tl.int32)
    
    # Adjust i to ensure we're in correct triangular position
    i_start = i * (i - 1) // 2
    i_too_small = i_start > valid_indices
    i = tl.where(i_too_small, i - 1, i)
    
    # Recompute with corrected i
    i_start = i * (i - 1) // 2
    j = valid_indices - i_start
    
    # Bounds checking
    valid_coords = mask & (i < 256) & (j < i) & (j >= 0)
    
    # Calculate offsets for aa[j][i], bb[i][j], and aa[i][j]
    aa_ji_offset = j * 256 + i
    bb_ij_offset = i * 256 + j  
    aa_ij_offset = i * 256 + j
    
    # Load values
    aa_ji_val = tl.load(aa_ptr + aa_ji_offset, mask=valid_coords, other=0.0)
    bb_ij_val = tl.load(bb_ptr + bb_ij_offset, mask=valid_coords, other=0.0)
    
    # Compute result
    result = aa_ji_val + bb_ij_val
    
    # Store result
    tl.store(aa_ptr + aa_ij_offset, result, mask=valid_coords)

def s114_triton(aa, bb):
    n = aa.shape[0]
    n_elements = n * (n - 1) // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa