import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Sequential loop over i dimension
    for i in range(LEN_2D):
        # 2D array operation: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        # Calculate linear indices for 2D arrays (row-major: aa[j][i] = j * LEN_2D + i)
        aa_indices = j_offsets * LEN_2D + i
        
        # Load values from 2D arrays
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + aa_indices, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + aa_indices, mask=j_mask, other=0.0)
        
        # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_result = aa_vals + bb_vals * cc_vals
        
        # Store result back to aa
        tl.store(aa_ptr + aa_indices, aa_result, mask=j_mask)
        
        # 1D array operation: a[i] = b[i] + c[i] * d[i] (only first thread)
        if pid_j == 0:
            first_thread = j_offsets[0]
            if first_thread == 0:
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                
                a_result = b_val + c_val * d_val
                tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    # Get dimensions from input tensors
    LEN_2D = aa.shape[0]
    
    # Set block size
    BLOCK_SIZE = 64
    
    # Calculate grid size for j dimension
    grid_j = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Launch kernel
    s2275_kernel[(grid_j,)](
        a, aa, b, bb, c, cc, d,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )