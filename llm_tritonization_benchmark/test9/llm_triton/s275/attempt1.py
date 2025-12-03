import torch
import triton
import triton.language as tl

@triton.jit
def s275_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get column index
    col_idx = tl.program_id(0)
    
    if col_idx >= LEN_2D:
        return
    
    # Check condition aa[0][col_idx] > 0
    aa_0_offset = col_idx
    condition_val = tl.load(aa_ptr + aa_0_offset)
    
    if condition_val > 0.0:
        # Process rows 1 to LEN_2D-1 in blocks
        row_offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(1, LEN_2D, BLOCK_SIZE):
            current_rows = block_start + row_offsets
            mask = (current_rows < LEN_2D) & (current_rows >= 1)
            
            if tl.sum(mask.to(tl.int32)) == 0:
                break
            
            # Calculate offsets for current positions aa[current_rows][col_idx]
            current_offsets = current_rows * LEN_2D + col_idx
            prev_offsets = (current_rows - 1) * LEN_2D + col_idx
            
            # Load previous values aa[current_rows-1][col_idx]
            prev_vals = tl.load(aa_ptr + prev_offsets, mask=mask)
            
            # Load bb[current_rows][col_idx]
            bb_vals = tl.load(bb_ptr + current_offsets, mask=mask)
            
            # Load cc[current_rows][col_idx]
            cc_vals = tl.load(cc_ptr + current_offsets, mask=mask)
            
            # Compute aa[current_rows][col_idx] = aa[current_rows-1][col_idx] + bb[current_rows][col_idx] * cc[current_rows][col_idx]
            result = prev_vals + bb_vals * cc_vals
            
            # Store result
            tl.store(aa_ptr + current_offsets, result, mask=mask)

def s275_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with one thread block per column
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s275_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa