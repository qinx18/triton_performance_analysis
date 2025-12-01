import triton
import triton.language as tl
import torch

@triton.jit
def s235_kernel(
    a_ptr, b_ptr, c_ptr, aa_ptr, bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for the i dimension
    pid = tl.program_id(axis=0)
    
    # Calculate the range of i indices this block will handle
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LEN_2D
    
    # Load b[i] and c[i] for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Load and update a[i] = a[i] + b[i] * c[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    a_vals = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # For each i in this block, compute the inner j loop sequentially
    for i_idx in range(BLOCK_SIZE):
        i = block_start + i_idx
        if i >= LEN_2D:
            break
            
        # Get the updated a[i] value
        a_i = tl.load(a_ptr + i)
        
        # Sequential loop over j from 1 to LEN_2D-1
        for j in range(1, LEN_2D):
            # aa[j][i] = aa[j-1][i] + bb[j][i] * a[i]
            aa_prev = tl.load(aa_ptr + (j-1) * LEN_2D + i)
            bb_ji = tl.load(bb_ptr + j * LEN_2D + i)
            aa_new = aa_prev + bb_ji * a_i
            tl.store(aa_ptr + j * LEN_2D + i, aa_new)

def s235_triton(a, b, c, aa, bb):
    LEN_2D = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid_size = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s235_kernel[(grid_size,)](
        a, b, c, aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )