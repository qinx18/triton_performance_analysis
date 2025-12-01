import triton
import triton.language as tl
import torch

@triton.jit
def s2275_kernel(
    aa_ptr, bb_ptr, cc_ptr, a_ptr, b_ptr, c_ptr, d_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0)
    
    if i < LEN_2D:
        # Inner loop over j for 2D arrays
        for j in range(0, LEN_2D, BLOCK_SIZE):
            j_offsets = j + tl.arange(0, BLOCK_SIZE)
            mask = j_offsets < LEN_2D
            
            # Calculate 2D indices: aa[j][i] -> aa[j * LEN_2D + i]
            aa_indices = j_offsets * LEN_2D + i
            bb_indices = j_offsets * LEN_2D + i
            cc_indices = j_offsets * LEN_2D + i
            
            # Load values
            aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
            bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
            cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
            
            # Compute: aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
            result = aa_vals + bb_vals * cc_vals
            
            # Store result
            tl.store(aa_ptr + aa_indices, result, mask=mask)
        
        # 1D array computation: a[i] = b[i] + c[i] * d[i]
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        result_1d = b_val + c_val * d_val
        tl.store(a_ptr + i, result_1d)

def s2275_triton(aa, bb, cc, a, b, c, d):
    LEN_2D = aa.shape[0]
    
    # Ensure arrays are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous()
    cc = cc.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    BLOCK_SIZE = 32
    grid = (LEN_2D,)
    
    s2275_kernel[grid](
        aa, bb, cc, a, b, c, d,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )