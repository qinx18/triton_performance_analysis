import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(
    bb_ptr, cc_ptr, aa_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which j we're processing
    j = pid
    
    if j >= LEN_2D:
        return
    
    # Process elements from i=j to i=LEN_2D-1
    num_elements = LEN_2D - j
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for block_start in range(0, num_elements, BLOCK_SIZE):
        # Calculate actual i indices (starting from j)
        i_offsets = j + block_start + offsets
        
        # Create mask for valid elements
        mask = (block_start + offsets < num_elements)
        
        # Calculate linear indices for 2D arrays
        linear_indices = i_offsets * LEN_2D + j
        
        # Load bb[i][j] and cc[i][j]
        bb_vals = tl.load(bb_ptr + linear_indices, mask=mask)
        cc_vals = tl.load(cc_ptr + linear_indices, mask=mask)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + linear_indices, result, mask=mask)

def s1232_triton(bb, cc, aa):
    LEN_2D = bb.shape[0]
    
    # Ensure arrays are contiguous
    bb = bb.contiguous()
    cc = cc.contiguous()
    aa = aa.contiguous()
    
    # Launch kernel with one thread block per j
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        bb, cc, aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa