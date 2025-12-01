import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the first value greater than threshold
    # Each program handles one block of elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find first element > t in this block
    condition = a_vals > t
    
    # Create a mask for valid indices within bounds
    valid_mask = mask & condition
    
    # If any element in this block satisfies condition
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        # Find the first occurrence
        # Create sequence numbers for each element in block
        seq = tl.arange(0, BLOCK_SIZE)
        
        # Set sequence to large number where condition is false
        seq = tl.where(valid_mask, seq, BLOCK_SIZE)
        
        # Find minimum sequence number (first occurrence)
        min_seq = tl.min(seq)
        
        # Check if this block has the first occurrence
        if min_seq < BLOCK_SIZE:
            first_idx = block_start + min_seq
            first_val = tl.load(a_ptr + first_idx)
            
            # Atomically update result if this is earlier than current result
            old_idx = tl.atomic_cas(result_ptr, -2, first_idx)
            if old_idx == -2:
                # We were first to find a result
                tl.store(result_ptr + 1, first_val)
            elif first_idx < old_idx:
                # Our index is earlier, try to update
                success = False
                while not success and first_idx < tl.load(result_ptr):
                    old_idx = tl.atomic_cas(result_ptr, old_idx, first_idx)
                    if old_idx == first_idx or old_idx > first_idx:
                        tl.store(result_ptr + 1, first_val)
                        success = True

def s332_triton(a, t_val):
    n_elements = a.numel()
    
    # Result tensor: [index, value]
    result = torch.tensor([-2, -1.0], dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s332_kernel[grid](
        a, t_val, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    index = int(result[0].item())
    value = result[1].item()
    chksum = value + float(index)
    
    return value