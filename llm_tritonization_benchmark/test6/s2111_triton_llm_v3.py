import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel implements a wavefront/anti-diagonal parallelization
    # Each thread block processes one anti-diagonal
    
    diag_id = tl.program_id(0)
    
    # For diagonal d, we have elements (j,i) where j+i = d+2
    # j ranges from 1 to min(d+1, LEN_2D-1)
    # i = d+2-j, and i must be >= 1 and < LEN_2D
    
    diagonal = diag_id + 2
    max_j = tl.minimum(diagonal + 1, LEN_2D - 1)
    
    if diagonal >= 2 * LEN_2D - 2:
        return
    
    # Each thread in the block processes one element of the diagonal
    tid = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    j = tid + 1
    i = diagonal - j
    
    # Bounds check
    valid = (j >= 1) & (j < LEN_2D) & (i >= 1) & (i < LEN_2D) & (j < max_j)
    
    # Load values
    current_offset = j * LEN_2D + i
    left_offset = j * LEN_2D + (i - 1)
    above_offset = (j - 1) * LEN_2D + i
    
    left_val = tl.load(aa_ptr + left_offset, mask=valid, other=0.0)
    above_val = tl.load(aa_ptr + above_offset, mask=valid, other=0.0)
    
    # Compute new value
    new_val = (left_val + above_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_offset, new_val, mask=valid)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Create a copy to avoid read-after-write dependencies
    aa_work = aa.clone()
    
    # Process each anti-diagonal sequentially
    num_diagonals = 2 * LEN_2D - 4  # diagonals from 2 to 2*LEN_2D-3
    
    for diag in range(num_diagonals):
        diagonal = diag + 2
        max_elements = min(diagonal - 1, 2 * LEN_2D - diagonal - 1, LEN_2D - 1)
        
        if max_elements <= 0:
            continue
            
        num_blocks = triton.cdiv(max_elements, BLOCK_SIZE)
        
        s2111_kernel[(1, num_blocks)](
            aa_work,
            LEN_2D=LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    aa.copy_(aa_work)