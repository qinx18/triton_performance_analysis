import torch
import triton
import triton.language as tl

@triton.jit
def s118_kernel(
    a_ptr,
    bb_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s118 computation.
    Each program handles one value of i, processing all j values sequentially.
    """
    pid = tl.program_id(0)
    i = pid + 1  # i starts from 1
    
    if i >= n:
        return
    
    # Load current a[i] value
    a_i = tl.load(a_ptr + i)
    
    # Sequential loop over j from 0 to i-1
    for j in range(i):
        # Load bb[j, i] and a[i-j-1]
        bb_ji = tl.load(bb_ptr + j * n + i)
        a_idx = i - j - 1
        a_val = tl.load(a_ptr + a_idx)
        
        # Accumulate: a[i] += bb[j][i] * a[i-j-1]
        a_i += bb_ji * a_val
    
    # Store updated a[i]
    tl.store(a_ptr + i, a_i)

def s118_triton(a, bb):
    """
    Triton implementation of TSVC s118 function.
    
    Optimizations:
    - Parallelizes over i dimension (outer loop)
    - Sequential processing of j loop to maintain dependencies
    - Coalesced memory access patterns for bb matrix
    - Direct in-place updates to minimize memory traffic
    """
    a = a.contiguous()
    bb = bb.contiguous()
    
    n = a.size(0)
    
    if n <= 1:
        return a
    
    # Launch kernel with one thread per i value (excluding i=0)
    num_programs = n - 1
    BLOCK_SIZE = 128
    
    grid = (triton.cdiv(num_programs, 1),)
    
    s118_kernel[grid](
        a_ptr=a,
        bb_ptr=bb,
        n=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a