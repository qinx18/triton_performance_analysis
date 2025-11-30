import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for the i dimension
    pid_i = tl.program_id(0)
    
    # Check bounds for i
    if pid_i >= LEN_2D:
        return
    
    # Process the inner j loop for this i
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        # Calculate j indices
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < LEN_2D
        
        # Calculate 2D array indices: aa[j][i] -> aa[j * LEN_2D + i]
        aa_indices = j_offsets * LEN_2D + pid_i
        bb_indices = j_offsets * LEN_2D + pid_i
        cc_indices = j_offsets * LEN_2D + pid_i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask)
        
        # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        result = aa_vals + bb_vals * cc_vals
        
        # Store result
        tl.store(aa_ptr + aa_indices, result, mask=j_mask)
    
    # Compute: a[i] = b[i] + c[i] * d[i]
    b_val = tl.load(b_ptr + pid_i)
    c_val = tl.load(c_ptr + pid_i)
    d_val = tl.load(d_ptr + pid_i)
    
    a_result = b_val + c_val * d_val
    tl.store(a_ptr + pid_i, a_result)

def s2275_triton(aa, bb, cc, a, b, c, d):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per i value
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )