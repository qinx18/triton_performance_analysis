import triton
import triton.language as tl

@triton.jit
def s1232_kernel(aa, bb, cc, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j in range(LEN_2D):
        i_indices = pid * BLOCK_SIZE + i_offsets
        
        mask = (i_indices < LEN_2D) & (i_indices >= j)
        
        flat_indices = i_indices * LEN_2D + j
        
        bb_vals = tl.load(bb + flat_indices, mask=mask)
        cc_vals = tl.load(cc + flat_indices, mask=mask)
        result = bb_vals + cc_vals
        tl.store(aa + flat_indices, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    s1232_kernel[grid](aa, bb, cc, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)