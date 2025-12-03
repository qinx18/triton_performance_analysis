import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for the outer loop dimension
    pid = tl.program_id(0)
    
    # Calculate which i we're processing
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Process the inner j loop for this i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        j_mask = current_j_offsets < LEN_2D
        
        # Calculate 2D indices for aa[j][i], bb[j][i], cc[j][i]
        aa_indices = current_j_offsets * LEN_2D + i
        bb_indices = current_j_offsets * LEN_2D + i
        cc_indices = current_j_offsets * LEN_2D + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask)
        
        # Compute aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store back
        tl.store(aa_ptr + aa_indices, result, mask=j_mask)
    
    # Compute a[i] = b[i] + c[i] * d[i]
    b_val = tl.load(b_ptr + i)
    c_val = tl.load(c_ptr + i)
    d_val = tl.load(d_ptr + i)
    
    result_a = b_val + c_val * d_val
    tl.store(a_ptr + i, result_a)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )