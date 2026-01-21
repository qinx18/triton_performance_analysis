import triton
import triton.language as tl

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, i, LEN_2D: tl.constexpr):
    k_start = i * LEN_2D + 1
    
    for j in range(1, LEN_2D):
        bb_prev_offset = (j - 1) * LEN_2D + i
        bb_curr_offset = j * LEN_2D + i
        cc_offset = j * LEN_2D + i
        flat_offset = k_start + (j - 1) - 1
        
        bb_prev_val = tl.load(bb_ptr + bb_prev_offset)
        cc_val = tl.load(cc_ptr + cc_offset)
        flat_val = tl.load(flat_2d_array_ptr + flat_offset)
        
        result = bb_prev_val + flat_val * cc_val
        tl.store(bb_ptr + bb_curr_offset, result)

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    for i in range(LEN_2D):
        s126_kernel[(1,)](bb, cc, flat_2d_array, i, LEN_2D=LEN_2D)