import triton
import triton.language as tl
import torch

@triton.jit
def s31111_kernel(a_ptr, sum_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        sum_val = 0.0
        
        # Process each group of 4 elements: indices 0-31
        for start_idx in range(8):  # 8 groups: 0,4,8,12,16,20,24,28
            base_offset = start_idx * 4
            if base_offset < N:
                # Load up to 4 elements starting at base_offset
                offsets = base_offset + tl.arange(0, 4)
                mask = offsets < N
                vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
                sum_val += tl.sum(vals)
        
        tl.store(sum_ptr, sum_val)

def s31111_triton(a):
    N = a.shape[0]
    
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    
    s31111_kernel[grid](
        a, sum_tensor, N
    )
    
    return sum_tensor.item()