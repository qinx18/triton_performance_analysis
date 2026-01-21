import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    for i in range(LEN_2D):
        # Update aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_offsets = j_idx * LEN_2D + i
        bb_offsets = j_idx * LEN_2D + i  
        cc_offsets = j_idx * LEN_2D + i
        
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
        cc_vals = tl.load(cc_ptr + cc_offsets, mask=j_mask, other=0.0)
        
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_offsets, result, mask=j_mask)
        
        # Update a[i] = b[i] + c[i] * d[i] (only once per i)
        if pid == 0:
            first_idx = j_offsets[0]
            if first_idx == 0:
                a_val = tl.load(a_ptr + i)
                b_val = tl.load(b_ptr + i)
                c_val = tl.load(c_ptr + i)
                d_val = tl.load(d_ptr + i)
                tl.store(a_ptr + i, b_val + c_val * d_val)

def s2275_triton(a, aa, b, bb, c, cc, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
    
    s2275_kernel[grid](
        a, aa, b, bb, c, cc, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )