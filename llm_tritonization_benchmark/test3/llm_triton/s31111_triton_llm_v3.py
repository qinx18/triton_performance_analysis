import torch
import triton
import triton.language as tl

@triton.jit
def test_func(ptr):
    return tl.load(ptr)

@triton.jit
def s31111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values at specific offsets
    val0 = tl.load(a_ptr + 0, mask=pid == 0)
    val1 = tl.load(a_ptr + 4, mask=pid == 0)
    val2 = tl.load(a_ptr + 8, mask=pid == 0)
    val3 = tl.load(a_ptr + 12, mask=pid == 0)
    val4 = tl.load(a_ptr + 16, mask=pid == 0)
    val5 = tl.load(a_ptr + 20, mask=pid == 0)
    val6 = tl.load(a_ptr + 24, mask=pid == 0)
    val7 = tl.load(a_ptr + 28, mask=pid == 0)
    
    # Sum reduction
    local_sum = val0 + val1 + val2 + val3 + val4 + val5 + val6 + val7
    
    # Store result only from first block
    if pid == 0:
        tl.store(result_ptr, local_sum)

def s31111_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s31111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()