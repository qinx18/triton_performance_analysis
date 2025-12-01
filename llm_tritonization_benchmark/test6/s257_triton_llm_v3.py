import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get block of i indices (starting from 1)
    block_start = tl.program_id(0) * BLOCK_SIZE + 1
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    i_mask = i_indices < LEN_2D
    
    # Process all j values for this block of i indices
    for j in range(LEN_2D):
        # Calculate array indices
        i_offsets = i_indices
        i_prev_offsets = i_indices - 1
        aa_ji_offsets = j * LEN_2D + i_indices
        bb_ji_offsets = j * LEN_2D + i_indices
        
        # Load values
        a_prev = tl.load(a_copy_ptr + i_prev_offsets, mask=i_mask)
        aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=i_mask)
        bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=i_mask)
        
        # Compute: a[i] = aa[j][i] - a[i-1]
        a_new = aa_ji - a_prev
        
        # Store a[i]
        tl.store(a_ptr + i_offsets, a_new, mask=i_mask)
        
        # Compute: aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_ji
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_ji_offsets, aa_new, mask=i_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)  # -1 because we start from i=1
    
    grid = (num_blocks,)
    
    s257_kernel[grid](
        a, a_copy, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )