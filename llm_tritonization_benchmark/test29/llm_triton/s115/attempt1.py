import triton
import triton.language as tl

@triton.jit
def s115_kernel(a_ptr, aa_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    
    for j in range(len_2d):
        # Only process i values where i > j
        i_mask = (i_idx < len_2d) & (i_idx > j)
        
        # Load a[j] (scalar broadcast)
        a_j = tl.load(a_ptr + j)
        
        # Load aa[j][i] for valid i indices
        aa_ji = tl.load(aa_ptr + j * len_2d + i_idx, mask=i_mask, other=0.0)
        
        # Load a[i] for valid i indices
        a_i = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        # Compute a[i] -= aa[j][i] * a[j]
        result = a_i - aa_ji * a_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s115_kernel[grid](
        a, aa, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )