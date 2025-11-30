import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(
    a_ptr, b_ptr,
    n1, n3,
    LEN_1D,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the number of iterations in the original loop
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate which iteration this thread handles
    pid = tl.program_id(axis=0)
    
    if pid >= num_iters:
        return
    
    # Calculate i, j, k values for this iteration
    i = (n1 - 1) + pid * n3
    j = 1
    k = j * (pid + 1)  # k accumulates j for each iteration
    
    # Bounds check
    if i < LEN_1D:
        # Load and update: a[i] += b[LEN_1D - k]
        b_idx = LEN_1D - k
        if b_idx >= 0 and b_idx < LEN_1D:
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + b_idx)
            tl.store(a_ptr + i, a_val + b_val)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations
    num_iters = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iters <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iters, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b,
        n1, n3,
        LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )