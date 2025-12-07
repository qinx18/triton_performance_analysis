import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate initial k: k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1
    # Simplifies to: k = i * (i+1) / 2 + i = i * (i+3) / 2
    k = i * (i + 3) // 2
    
    # Process inner loop from j=i to LEN_2D-1 in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(i, LEN_2D, BLOCK_SIZE):
        j_vals = j_start + j_offsets
        mask = (j_vals < LEN_2D) & (j_vals >= i)
        
        if tl.sum(mask.to(tl.int32)) == 0:
            break
        
        # Calculate k indices for this block
        k_vals = k + tl.cumsum(j_vals, axis=0) - tl.cumsum(tl.full((BLOCK_SIZE,), i, tl.int32), axis=0) + i
        
        # Load bb[j][i] values
        bb_indices = j_vals * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
        
        # Load current flat_2d_array values
        flat_vals = tl.load(flat_2d_array_ptr + k_vals, mask=mask, other=0.0)
        
        # Add and store back
        new_vals = flat_vals + bb_vals
        tl.store(flat_2d_array_ptr + k_vals, new_vals, mask=mask)
        
        # Update k for next iteration
        if j_start + BLOCK_SIZE < LEN_2D:
            last_valid_j = min(j_start + BLOCK_SIZE - 1, LEN_2D - 1)
            k += last_valid_j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 64
    
    grid = (LEN_2D,)
    
    s141_kernel[grid](
        bb, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )