import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, i_idx: tl.constexpr):
    # Process column i_idx
    k = i_idx * LEN_2D + 1
    
    # Sequential processing for j from 1 to LEN_2D-1
    for j in range(1, LEN_2D):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + i_idx
        bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i_idx
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev_val + flat_val * cc_val
        
        # Store result
        bb_offset = j * LEN_2D + i_idx
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch one thread per column (i dimension)
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        i_idx=triton.language.program_id(0)
    )