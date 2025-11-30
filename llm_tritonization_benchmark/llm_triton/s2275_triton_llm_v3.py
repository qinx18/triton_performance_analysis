import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
):
    # Get program IDs
    pid_i = tl.program_id(0)
    
    # Calculate i range for this block
    i_start = pid_i * BLOCK_SIZE_I
    i_end = tl.minimum(i_start + BLOCK_SIZE_I, LEN_2D)
    
    # Process each i in this block
    for i_offset in range(BLOCK_SIZE_I):
        i = i_start + i_offset
        if i >= LEN_2D:
            break
            
        # Inner j loop - process all j for this i
        for j_start in range(0, LEN_2D, BLOCK_SIZE_J):
            j_end = tl.minimum(j_start + BLOCK_SIZE_J, LEN_2D)
            j_offsets = tl.arange(0, BLOCK_SIZE_J) + j_start
            j_mask = j_offsets < LEN_2D
            
            # Calculate 2D array indices: aa[j][i] -> j * LEN_2D + i
            aa_indices = j_offsets * LEN_2D + i
            bb_indices = j_offsets * LEN_2D + i
            cc_indices = j_offsets * LEN_2D + i
            
            # Load values
            aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
            bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
            cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask)
            
            # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
            result = aa_vals + bb_vals * cc_vals
            
            # Store back
            tl.store(aa_ptr + aa_indices, result, mask=j_mask)
        
        # After processing all j for this i, update 1D arrays
        # a[i] = b[i] + c[i] * d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_result = b_val + c_val * d_val
        tl.store(a_ptr + i, a_result)

def s2275_triton(aa, bb, cc, a, b, c, d):
    LEN_2D = aa.shape[0]
    
    # Launch configuration
    BLOCK_SIZE_I = 32
    BLOCK_SIZE_J = 64
    
    num_blocks_i = triton.cdiv(LEN_2D, BLOCK_SIZE_I)
    
    # Launch kernel
    s2275_kernel[(num_blocks_i,)](
        aa, bb, cc, a, b, c, d,
        LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J,
    )