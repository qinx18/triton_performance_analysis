import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(
    a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process j from 1 to LEN_2D-1 in blocks
    for block_start in range(1, LEN_2D, BLOCK_SIZE):
        j_offsets = block_start + offsets
        mask = j_offsets < LEN_2D
        
        # Load previous values (j-1)
        prev_offsets = j_offsets - 1
        prev_mask = (prev_offsets >= 0) & (prev_offsets < LEN_2D) & mask
        a_prev = tl.load(a_copy_ptr + prev_offsets, mask=prev_mask, other=0.0)
        
        # Compute a[j] = 1.0 - a[j-1]
        a_vals = 1.0 - a_prev
        
        # Store a[j]
        tl.store(a_ptr + j_offsets, a_vals, mask=mask)
        
        # Load d[j] and bb[j][i]
        d_vals = tl.load(d_ptr + j_offsets, mask=mask, other=0.0)
        bb_offsets = j_offsets * LEN_2D + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_vals + bb_vals * d_vals
        
        # Store aa[j][i]
        aa_offsets = j_offsets * LEN_2D + i
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = a.shape[0]
    BLOCK_SIZE = 32
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    # Launch kernel for each i
    grid = (LEN_2D,)
    
    s256_kernel[grid](
        a, a_copy, aa, bb, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )