import triton
import triton.language as tl

@triton.jit
def s233_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = pid * BLOCK_SIZE + i_offsets + 1
    i_mask = i_indices < LEN_2D
    
    for j in range(1, LEN_2D):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * LEN_2D + i_indices
        aa_write_offsets = j * LEN_2D + i_indices
        cc_offsets = j * LEN_2D + i_indices
        
        aa_prev = tl.load(aa_ptr + aa_read_offsets, mask=i_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=i_mask, other=0.0)
        aa_new = aa_prev + cc_vals
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=i_mask)

@triton.jit
def s233_bb_kernel(bb_ptr, cc_ptr, LEN_2D: tl.constexpr):
    for i in range(1, LEN_2D):
        for j in range(1, LEN_2D):
            bb_read_offset = j * LEN_2D + (i - 1)
            bb_write_offset = j * LEN_2D + i
            cc_offset = j * LEN_2D + i
            
            bb_prev = tl.load(bb_ptr + bb_read_offset)
            cc_val = tl.load(cc_ptr + cc_offset)
            bb_new = bb_prev + cc_val
            tl.store(bb_ptr + bb_write_offset, bb_new)

def s233_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # First kernel for aa computation - parallelize over i
    grid = (triton.cdiv(LEN_2D - 1, BLOCK_SIZE),)
    s233_kernel[grid](aa, bb, cc, LEN_2D, BLOCK_SIZE)
    
    # Second kernel for bb computation - sequential
    s233_bb_kernel[(1,)](bb, cc, LEN_2D)