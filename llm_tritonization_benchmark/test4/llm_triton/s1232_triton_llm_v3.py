import triton
import triton.language as tl
import torch

@triton.jit
def s1232_kernel(aa_ptr, bb_ptr, cc_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get current j value from program id
    j = tl.program_id(0)
    
    if j >= LEN_2D:
        return
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements from i = j to i = LEN_2D-1
    num_elements = LEN_2D - j
    
    # Process in blocks
    for block_start in range(0, num_elements, BLOCK_SIZE):
        # Calculate actual i indices
        i_offsets = j + block_start + offsets
        
        # Create mask for valid elements
        mask = (block_start + offsets < num_elements) & (i_offsets < LEN_2D)
        
        # Calculate memory addresses
        addr_offsets = i_offsets * LEN_2D + j
        
        # Load values
        bb_vals = tl.load(bb_ptr + addr_offsets, mask=mask)
        cc_vals = tl.load(cc_ptr + addr_offsets, mask=mask)
        
        # Compute result
        result = bb_vals + cc_vals
        
        # Store result
        tl.store(aa_ptr + addr_offsets, result, mask=mask)

def s1232_triton(aa, bb, cc):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread block per j value
    grid = (LEN_2D,)
    
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )