import torch
import triton
import triton.language as tl

@triton.jit
def s235_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    aa_ptr,
    aa_copy_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the i dimension
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load a[i], b[i], c[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Update a[i] += b[i] * c[i]
    a_new = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Inner j loop: aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
    # This must be sequential in j dimension
    for j in range(1, LEN_2D):
        # Calculate offsets for 2D arrays
        j_prev_offsets = (j - 1) * LEN_2D + offsets
        j_offsets = j * LEN_2D + offsets
        
        # Load aa[j-1][i] from read-only copy
        aa_prev_vals = tl.load(aa_copy_ptr + j_prev_offsets, mask=mask)
        
        # Load bb[j][i]
        bb_vals = tl.load(bb_ptr + j_offsets, mask=mask)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
        aa_new = aa_prev_vals + bb_vals * a_new
        
        # Store to original aa array
        tl.store(aa_ptr + j_offsets, aa_new, mask=mask)
        
        # Update copy for next iteration
        tl.store(aa_copy_ptr + j_offsets, aa_new, mask=mask)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 64
    
    # Create read-only copy of aa to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Calculate grid size
    grid = ((LEN_2D + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Launch kernel
    s235_kernel[grid](
        a, b, c, aa, aa_copy, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )