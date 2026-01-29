import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute offsets for i dimension - define once outside loop
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_start = pid * BLOCK_SIZE
    i_idx = i_start + i_offsets
    
    # Sequential loop over j dimension
    for j in range(LEN_2D):
        # Create mask for valid i values (i >= j and i < LEN_2D)
        mask_valid_i = i_idx < LEN_2D
        mask_triangular = i_idx >= j
        mask = mask_valid_i & mask_triangular
        
        # Compute memory addresses for this (i, j) block
        bb_addrs = bb_ptr + i_idx * LEN_2D + j
        cc_addrs = cc_ptr + i_idx * LEN_2D + j
        aa_addrs = aa_ptr + i_idx * LEN_2D + j
        
        # Load bb[i][j] and cc[i][j] values with mask
        bb_vals = tl.load(bb_addrs, mask=mask, other=0.0)
        cc_vals = tl.load(cc_addrs, mask=mask, other=0.0)
        
        # Compute aa[i][j] = bb[i][j] + cc[i][j]
        aa_vals = bb_vals + cc_vals
        
        # Store result only for valid triangular elements
        tl.store(aa_addrs, aa_vals, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Calculate grid size for i dimension parallelization
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )