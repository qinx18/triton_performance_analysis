import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    
    # Calculate i index
    i = pid_i
    
    if i < LEN_2D:
        # Process inner j loop for 2D arrays
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < LEN_2D
            
            # Calculate 2D array offsets (aa[j][i] means j*LEN_2D + i)
            aa_offsets = j_indices * LEN_2D + i
            bb_offsets = j_indices * LEN_2D + i
            cc_offsets = j_indices * LEN_2D + i
            
            # Load values
            aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=0.0)
            bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
            cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
            
            # Compute aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
            result_vals = aa_vals + bb_vals * cc_vals
            
            # Store results
            tl.store(aa_ptr + aa_offsets, result_vals, mask=j_mask)
        
        # Process 1D arrays: a[i] = b[i] + c[i] * d[i]
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        result_a = b_val + c_val * d_val
        tl.store(a_ptr + i, result_a)

def s2275_triton(a, b, c, d, aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread per i
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, b, c, d,
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )