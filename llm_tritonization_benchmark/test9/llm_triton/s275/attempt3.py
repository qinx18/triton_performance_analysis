import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
):
    # Get column index
    col_idx = tl.program_id(0)
    
    if col_idx >= LEN_2D:
        return
    
    # Check condition aa[0][col_idx] > 0
    aa_0_offset = col_idx
    condition_val = tl.load(aa_ptr + aa_0_offset)
    
    if condition_val > 0.0:
        # Process rows 1 to LEN_2D-1 sequentially
        for j in range(1, LEN_2D):
            # Calculate offsets
            current_offset = j * LEN_2D + col_idx
            prev_offset = (j - 1) * LEN_2D + col_idx
            
            # Load values
            prev_val = tl.load(aa_ptr + prev_offset)
            bb_val = tl.load(bb_ptr + current_offset)
            cc_val = tl.load(cc_ptr + current_offset)
            
            # Compute aa[j][col_idx] = aa[j-1][col_idx] + bb[j][col_idx] * cc[j][col_idx]
            result = prev_val + bb_val * cc_val
            
            # Store result
            tl.store(aa_ptr + current_offset, result)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread per column
    grid = (LEN_2D,)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
    )
    
    return aa