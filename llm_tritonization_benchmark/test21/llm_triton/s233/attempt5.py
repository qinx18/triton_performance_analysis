import triton
import triton.language as tl
import torch

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, 
                LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets + 1  # Start from i=1
    i_mask = i_idx < LEN_2D
    
    # First nested loop: aa[j][i] = aa[j-1][i] + cc[j][i]
    for j in range(1, LEN_2D):
        # Read from aa[j-1][i]
        aa_prev_offsets = (j - 1) * LEN_2D + i_idx
        aa_prev = tl.load(aa_ptr + aa_prev_offsets, mask=i_mask, other=0.0)
        
        # Read from cc[j][i]
        cc_offsets = j * LEN_2D + i_idx
        cc_val = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        
        # Write to aa[j][i]
        aa_new = aa_prev + cc_val
        aa_write_offsets = j * LEN_2D + i_idx
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=i_mask)

@triton.jit
def s233_kernel_bb(bb_ptr, cc_ptr, 
                   LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets + 1  # Start from j=1
    j_mask = j_idx < LEN_2D
    
    # Second nested loop: bb[j][i] = bb[j][i-1] + cc[j][i]
    for i in range(1, LEN_2D):
        # Read from bb[j][i-1]
        bb_prev_offsets = j_idx * LEN_2D + (i - 1)
        bb_prev = tl.load(bb_ptr + bb_prev_offsets, mask=j_mask, other=0.0)
        
        # Read from cc[j][i]
        cc_offsets = j_idx * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        # Write to bb[j][i]
        bb_new = bb_prev + cc_val
        bb_write_offsets = j_idx * LEN_2D + i
        tl.store(bb_ptr + bb_write_offsets, bb_new, mask=j_mask)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # First loop: parallelize over i dimension
    i_size = LEN_2D - 1
    grid_i = (triton.cdiv(i_size, BLOCK_SIZE),)
    
    s233_kernel[grid_i](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second loop: parallelize over j dimension
    j_size = LEN_2D - 1
    grid_j = (triton.cdiv(j_size, BLOCK_SIZE),)
    
    s233_kernel_bb[grid_j](
        bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )