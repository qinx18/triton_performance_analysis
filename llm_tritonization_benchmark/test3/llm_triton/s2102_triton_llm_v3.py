import triton
import triton.language as tl
import torch

@triton.jit
def s2102_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer dimension (i)
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Zero out the entire column i
    for j_start in range(0, LEN_2D, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < LEN_2D
        
        # Calculate addresses: aa[j][i] = aa_ptr[j * LEN_2D + i]
        addrs = j_offsets * LEN_2D + i
        tl.store(aa_ptr + addrs, 0.0, mask=mask)
    
    # Set diagonal element aa[i][i] = 1.0
    diag_addr = i * LEN_2D + i
    tl.store(aa_ptr + diag_addr, 1.0)

def s2102_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    grid = (LEN_2D,)
    
    s2102_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )