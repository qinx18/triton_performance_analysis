import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, result_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    max_val = tl.load(aa_ptr)
    max_i = 0
    max_j = 0
    
    # Sequential loop over i dimension
    for i in range(N):
        # Parallel processing of j dimension
        for j_block_start in range(0, N, BLOCK_SIZE):
            j_offsets = j_block_start + offsets
            j_mask = j_offsets < N
            
            # Load values for current i, block of j
            load_offsets = i * N + j_offsets
            vals = tl.load(aa_ptr + load_offsets, mask=j_mask, other=float('-inf'))
            
            # Check if any value in this block is greater than current max
            greater_mask = vals > max_val
            
            if tl.sum(greater_mask.to(tl.int32)) > 0:
                # Find the position of maximum in this block
                local_max_idx = tl.argmax(vals, axis=0)
                local_max_val = tl.load(aa_ptr + i * N + j_block_start + local_max_idx)
                
                if local_max_val > max_val:
                    max_val = local_max_val
                    max_i = i
                    max_j = j_block_start + local_max_idx
    
    # Store result (only thread 0 of block 0)
    if pid == 0:
        result = max_val + max_i + 1 + max_j + 1
        tl.store(result_ptr, result)

def s3110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    
    grid = (1,)
    s3110_kernel[grid](aa, result, N=N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()