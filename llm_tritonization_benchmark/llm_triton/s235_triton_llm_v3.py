import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(
    a_ptr, b_ptr, c_ptr, aa_ptr, aa_copy_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get block ID for i dimension
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load values for a[i] += b[i] * c[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i]
    a_vals = a_vals + b_vals * c_vals
    
    # Store updated a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Inner j loop: aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
    # This must be done sequentially for each i in this block
    for block_idx in range(BLOCK_SIZE):
        if block_start + block_idx >= LEN_2D:
            break
            
        i = block_start + block_idx
        a_val = tl.load(a_ptr + i)  # Get updated a[i] value
        
        # Sequential loop over j from 1 to LEN_2D-1
        for j in range(1, LEN_2D):
            # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
            prev_offset = (j - 1) * LEN_2D + i
            curr_offset = j * LEN_2D + i
            bb_offset = j * LEN_2D + i
            
            aa_prev = tl.load(aa_copy_ptr + prev_offset)
            bb_val = tl.load(bb_ptr + bb_offset)
            
            result = aa_prev + bb_val * a_val
            tl.store(aa_ptr + curr_offset, result)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    
    # Create read-only copy of aa to handle WAR dependencies
    aa_copy = aa.clone()
    
    # Choose block size
    BLOCK_SIZE = 64
    
    # Calculate grid size
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s235_kernel[(grid_size,)](
        a, b, c, aa, aa_copy, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )