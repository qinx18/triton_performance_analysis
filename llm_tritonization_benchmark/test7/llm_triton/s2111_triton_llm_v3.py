import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr):
    # This kernel processes one diagonal at a time sequentially
    # Each diagonal contains elements where j+i = constant
    
    # Process diagonals sequentially from diagonal 2 to 2*(LEN_2D-1)
    for diag in range(2, 2 * LEN_2D):
        # For each diagonal, process all valid (j,i) pairs where j+i = diag
        # and 1 <= j < LEN_2D and 1 <= i < LEN_2D
        
        j_start = max(1, diag - (LEN_2D - 1))
        j_end = min(LEN_2D - 1, diag - 1)
        
        for j in range(j_start, j_end + 1):
            i = diag - j
            if 1 <= i < LEN_2D:
                # Load aa[j][i-1] and aa[j-1][i]
                left_val = tl.load(aa_ptr + j * LEN_2D + (i - 1))
                up_val = tl.load(aa_ptr + (j - 1) * LEN_2D + i)
                
                # Compute and store result
                result = (left_val + up_val) / 1.9
                tl.store(aa_ptr + j * LEN_2D + i, result)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with single thread since we need sequential processing
    grid = (1,)
    s2111_kernel[grid](aa, LEN_2D)
    
    return aa