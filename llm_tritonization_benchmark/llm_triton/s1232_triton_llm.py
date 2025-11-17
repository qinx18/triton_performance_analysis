import torch
import triton
import triton.language as tl

@triton.jit
def s1232_kernel(
    aa_ptr, bb_ptr, cc_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1232 - triangular matrix update
    Each program handles one element in the upper triangular matrix
    """
    pid = tl.program_id(0)
    
    # Convert linear program ID to (i, j) coordinates for upper triangular access
    # We need to find i, j such that i >= j and the linear index matches pid
    
    # Use quadratic formula to find row j from linear index
    # Total elements up to row j-1: j*(2*LEN_2D - j + 1)/2
    # Approximate j using quadratic formula
    discriminant = 2 * LEN_2D + 0.25 - 2 * pid
    j_approx = LEN_2D + 0.5 - tl.sqrt(discriminant)
    j = tl.maximum(0, tl.minimum(LEN_2D - 1, j_approx.to(tl.int32)))
    
    # Refine j to ensure we're in the correct row
    elements_before_j = j * (2 * LEN_2D - j - 1) // 2
    while elements_before_j > pid and j > 0:
        j -= 1
        elements_before_j = j * (2 * LEN_2D - j - 1) // 2
    
    while j < LEN_2D - 1:
        next_elements = (j + 1) * (2 * LEN_2D - j - 2) // 2
        if next_elements <= pid:
            j += 1
            elements_before_j = next_elements
        else:
            break
    
    # Calculate i from the remaining offset within row j
    i = j + (pid - elements_before_j)
    
    # Bounds check for valid triangular matrix access
    valid = (i < LEN_2D) & (j < LEN_2D) & (i >= j)
    
    if valid:
        # Calculate linear offset for 2D indexing
        offset = i * LEN_2D + j
        
        # Load values from bb and cc, compute sum, store to aa
        bb_val = tl.load(bb_ptr + offset)
        cc_val = tl.load(cc_ptr + offset)
        result = bb_val + cc_val
        tl.store(aa_ptr + offset, result)

def s1232_triton(aa, bb, cc):
    """
    Triton implementation of TSVC s1232
    Optimized for triangular matrix access pattern
    """
    aa = aa.contiguous()
    bb = bb.contiguous()  
    cc = cc.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Calculate total number of elements in upper triangular matrix (including diagonal)
    total_elements = LEN_2D * (LEN_2D + 1) // 2
    
    # Use smaller block size for better load balancing with triangular access
    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel with triangular matrix dimensions
    s1232_kernel[grid](
        aa, bb, cc,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa