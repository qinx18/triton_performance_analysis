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
    
    # Convert linear indices to (i, j) coordinates for j < i
    # Using inverse triangular mapping
    i_float = (1.0 + tl.sqrt(1.0 + 8.0 * indices.to(tl.float32))) / 2.0
    i = i_float.to(tl.int32)
    
    # Calculate j from the triangular offset
    triangular_base = i * (i - 1) // 2
    j = indices - triangular_base
    
    # Ensure valid triangular coordinates (j < i)
    valid_triangle = (j < i) & (j >= 0) & (i > 0)
    valid_coords = mask & valid_triangle & (i < 256) & (j < 256)
    
    # Calculate memory offsets
    aa_ji_offset = j * 256 + i  # aa[j][i]
    bb_ij_offset = i * 256 + j  # bb[i][j]  
    aa_ij_offset = i * 256 + j  # aa[i][j]
    
    # Load values
    aa_ji_val = tl.load(aa_ptr + aa_ji_offset, mask=valid_coords, other=0.0)
    bb_ij_val = tl.load(bb_ptr + bb_ij_offset, mask=valid_coords, other=0.0)
    
    # Compute result: aa[i][j] = aa[j][i] + bb[i][j]
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