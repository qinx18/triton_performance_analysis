import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each block handles one column i
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Initialize k for this column: k starts at 1 for each i, then increments by LEN_2D for each new i
    k = 1 + i * LEN_2D
    
    # Process all j values for this column sequentially
    for j in range(1, LEN_2D):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_offset = j * LEN_2D + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        bb_new = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * LEN_2D + i
        tl.store(bb_ptr + bb_offset, bb_new)
        
        # Increment k
        k += 1
    
    # After j loop finishes, increment k once more (the ++k after the j loop)
    # This is already handled correctly by the k initialization pattern

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with one block per column
    grid = (LEN_2D,)
    s126_kernel[grid](bb, cc, flat_2d_array, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)