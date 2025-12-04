import torch
import triton
import triton.language as tl

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel computes sum of elements at positions 0-31 using the test pattern
    # test() sums 4 consecutive elements starting at given positions: 0,4,8,12,16,20,24,28
    
    program_id = tl.program_id(0)
    if program_id != 0:  # Only one block needed for this computation
        return
    
    # Load elements needed for the computation (positions 0-31)
    offsets = tl.arange(0, 32)
    mask = offsets < 32
    vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute test(a) - sum elements 0,1,2,3
    test_a = vals[0] + vals[1] + vals[2] + vals[3]
    
    # Compute test(&a[4]) - sum elements 4,5,6,7
    test_a4 = vals[4] + vals[5] + vals[6] + vals[7]
    
    # Compute test(&a[8]) - sum elements 8,9,10,11
    test_a8 = vals[8] + vals[9] + vals[10] + vals[11]
    
    # Compute test(&a[12]) - sum elements 12,13,14,15
    test_a12 = vals[12] + vals[13] + vals[14] + vals[15]
    
    # Compute test(&a[16]) - sum elements 16,17,18,19
    test_a16 = vals[16] + vals[17] + vals[18] + vals[19]
    
    # Compute test(&a[20]) - sum elements 20,21,22,23
    test_a20 = vals[20] + vals[21] + vals[22] + vals[23]
    
    # Compute test(&a[24]) - sum elements 24,25,26,27
    test_a24 = vals[24] + vals[25] + vals[26] + vals[27]
    
    # Compute test(&a[28]) - sum elements 28,29,30,31
    test_a28 = vals[28] + vals[29] + vals[30] + vals[31]
    
    # Sum all test results
    total_sum = test_a + test_a4 + test_a8 + test_a12 + test_a16 + test_a20 + test_a24 + test_a28
    
    # Store result
    tl.store(sum_ptr, total_sum)

def s31111_triton(a):
    n_elements = a.numel()
    
    # Output tensor for the sum
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s31111_kernel[grid](
        a, sum_tensor, n_elements,
        BLOCK_SIZE=32
    )
    
    return sum_tensor[0].item()