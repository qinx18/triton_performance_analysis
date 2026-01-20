import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Each program handles one i
    pid = tl.program_id(0)
    i = pid
    
    if i >= LEN_2D:
        return
    
    # Initialize k for this i
    k = 1 + i * (LEN_2D - 1) + i
    
    # Process j from 1 to LEN_2D-1 sequentially
    for j in range(1, LEN_2D):
        # bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        bb_prev = tl.load(bb_ptr + (j-1) * LEN_2D + i)
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        
        result = bb_prev + flat_val * cc_val
        tl.store(bb_ptr + j * LEN_2D + i, result)
        
        k += 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(LEN_2D, 1),)
    
    s126_kernel[grid](bb, cc, flat_2d_array, LEN_2D, BLOCK_SIZE)