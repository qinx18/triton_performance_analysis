import triton
import triton.language as tl

@triton.jit
def s2275_kernel(a_ptr, aa_ptr, b_ptr, bb_ptr, c_ptr, cc_ptr, d_ptr, 
                 len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < len_2d
    
    for i in range(len_2d):
        # Compute 2D array indices for aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i]
        aa_indices = j_idx * len_2d + i
        bb_indices = j_idx * len_2d + i
        cc_indices = j_idx * len_2d + i
        
        # Load values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
        bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
        cc_vals = tl.load(cc_ptr + cc_indices, mask=j_mask)
        
        # Compute and store
        result = aa_vals + bb_vals * cc_vals
        tl.store(aa_ptr + aa_indices, result, mask=j_mask)
        
        # Only one thread updates a[i] to avoid race conditions
        if pid == 0 and j_offsets[0] == 0:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            a_result = b_val + c_val * d_val
            tl.store(a_ptr + i, a_result)

def s2275_triton(a, aa, b, bb, c, cc, d):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    s2275_kernel[grid](a, aa, b, bb, c, cc, d, len_2d, BLOCK_SIZE=BLOCK_SIZE)