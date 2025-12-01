import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID for outer loop (i dimension)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Process inner loop over j dimension in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = current_j < LEN_2D
        
        # Calculate 2D array indices: aa[j][i] means row j, column i
        indices = current_j * LEN_2D + i
        
        # Load aa[j][i], bb[j][i], cc[j][i]
        aa_vals = tl.load(aa_ptr + indices, mask=mask, other=0.0)
        bb_vals = tl.load(bb_ptr + indices, mask=mask, other=0.0)
        cc_vals = tl.load(cc_ptr + indices, mask=mask, other=0.0)
        
        # Compute aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store back to aa[j][i]
        tl.store(aa_ptr + indices, result, mask=mask)
    
    # Handle 1D array operation: a[i] = b[i] + c[i] * d[i]
    if i < LEN_2D:
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        result_1d = b_val + c_val * d_val
        tl.store(a_ptr + i, result_1d)

def s2275_triton(a, b, c, d, aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per i value
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, b, c, d,
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )