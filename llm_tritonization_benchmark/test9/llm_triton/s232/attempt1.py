import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential computation due to WAR dependency
    for j in range(1, LEN_2D):
        # Process row j in blocks
        for block_start in range(1, j + 1, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, j + 1)
            block_size = block_end - block_start
            
            if block_size > 0:
                # Create offsets for this block
                offsets = tl.arange(0, BLOCK_SIZE)
                i_offsets = block_start + offsets
                mask = (offsets < block_size) & (i_offsets <= j)
                
                # Compute indices
                aa_indices = j * LEN_2D + i_offsets
                aa_prev_indices = j * LEN_2D + (i_offsets - 1)
                bb_indices = j * LEN_2D + i_offsets
                
                # Load data
                aa_prev_vals = tl.load(aa_ptr + aa_prev_indices, mask=mask, other=0.0)
                bb_vals = tl.load(bb_ptr + bb_indices, mask=mask, other=0.0)
                
                # Compute: aa[j][i] = aa[j][i-1] * aa[j][i-1] + bb[j][i]
                result = aa_prev_vals * aa_prev_vals + bb_vals
                
                # Store result
                tl.store(aa_ptr + aa_indices, result, mask=mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Ensure tensors are contiguous
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    # Launch kernel with single thread block since computation is sequential
    s232_kernel[(1,)](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa