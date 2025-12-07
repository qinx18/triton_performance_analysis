import torch
import triton
import triton.language as tl

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Sequential processing of triangular region
    # Process elements where j < i in row-major order
    
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Convert linear indices to (i, j) coordinates for triangular region
    # We need to map linear index to valid (i, j) pairs where j < i
    
    # Load elements in blocks
    for idx in range(BLOCK_SIZE):
        linear_idx = block_start + idx
        if linear_idx >= n_elements:
            break
            
        # Convert linear index to triangular coordinates
        # For triangular region with j < i, we have sum from i=1 to N of i elements
        # So position k maps to some (i, j) where j < i
        
        # Find i by solving: k = i*(i-1)/2 + j where j < i
        # This means k < i*(i+1)/2
        i = 1
        cumsum = 0
        remaining = linear_idx
        
        # Find the row i
        while cumsum + i <= linear_idx:
            cumsum += i
            i += 1
        
        # j is the remainder
        j = linear_idx - cumsum
        
        # Bounds check
        if i < 256 and j < i:
            # Load aa[j][i] and bb[i][j]
            aa_ji_offset = j * 256 + i
            bb_ij_offset = i * 256 + j
            aa_ij_offset = i * 256 + j
            
            aa_ji_val = tl.load(aa_ptr + aa_ji_offset)
            bb_ij_val = tl.load(bb_ptr + bb_ij_offset)
            
            # Compute and store result
            result = aa_ji_val + bb_ij_val
            tl.store(aa_ptr + aa_ij_offset, result)

def s114_triton(aa, bb):
    # Calculate number of elements in triangular region (j < i)
    n = aa.shape[0]  # 256
    n_elements = n * (n - 1) // 2  # Number of elements where j < i
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s114_kernel[grid](
        aa, bb, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa