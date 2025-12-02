import torch
import triton
import triton.language as tl

@triton.jit
def s2275_kernel(
    a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID for i dimension
    pid = tl.program_id(axis=0)
    
    # Calculate i index
    i = pid
    
    if i < LEN_2D:
        # Process all j values for this i
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        # Process j dimension in blocks
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            current_j = j_start + j_offsets
            mask = current_j < LEN_2D
            
            # Calculate flat indices for aa[j][i], bb[j][i], cc[j][i]
            flat_indices = current_j * LEN_2D + i
            
            # Load values
            aa_vals = tl.load(aa_ptr + flat_indices, mask=mask)
            bb_vals = tl.load(bb_ptr + flat_indices, mask=mask)
            cc_vals = tl.load(cc_ptr + flat_indices, mask=mask)
            
            # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
            result = aa_vals + bb_vals * cc_vals
            
            # Store result
            tl.store(aa_ptr + flat_indices, result, mask=mask)
        
        # Compute: a[i] = b[i] + c[i] * d[i]
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        a_result = b_val + c_val * d_val
        tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Launch kernel with one thread block per i
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )