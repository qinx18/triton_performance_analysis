import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, b_ptr, c_ptr, d_ptr, aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer dimension (i)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Pre-define offsets for j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process the 2D arrays in blocks for j dimension
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_idx = j_start + j_offsets
        j_mask = j_idx < LEN_2D
        
        # Calculate memory offsets for aa[j][i], bb[j][i], cc[j][i]
        aa_offset = j_idx * LEN_2D + i
        bb_offset = j_idx * LEN_2D + i
        cc_offset = j_idx * LEN_2D + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_offset, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offset, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offset, mask=j_mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store back
        tl.store(aa_ptr + aa_offset, result, mask=j_mask)
    
    # Compute a[i] = b[i] + c[i] * d[i]
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    
    a_result = b_val + c_val * d_val
    tl.store(a_ptr + i, a_result)

def s2275_triton(a, b, c, d, aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, b, c, d, aa, bb, cc,
        LEN_2D, BLOCK_SIZE
    )