import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(b_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel implements stream compaction (packing positive values)
    # Each program processes one block of elements
    pid = tl.program_id(0)
    
    # Calculate the range of elements this program will process
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of b values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Find positive values in this block
    positive_mask = b_vals > 0.0
    
    # Count positive values in this block
    positive_count = tl.sum(positive_mask.to(tl.int32))
    
    # Use atomic add to get the starting position for this block's output
    output_start = tl.atomic_add(a_ptr - 1, positive_count)  # Use a[-1] as counter
    
    # Compact and store positive values
    write_idx = 0
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            val = tl.load(b_ptr + block_start + i)
            if val > 0.0:
                tl.store(a_ptr + output_start + write_idx, val)
                write_idx += 1

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Clear output array and use a[-1] as atomic counter
    a.fill_(0.0)
    
    # Use a separate counter tensor since we can't reliably use negative indexing
    counter = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Modified kernel that doesn't use negative indexing
    s341_kernel_v2[grid](
        b, a, counter, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )

@triton.jit
def s341_kernel_v2(b_ptr, a_ptr, counter_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process elements sequentially to maintain order (stream compaction requirement)
    for block_id in range(tl.num_programs(0)):
        if pid == 0:  # Only one program processes at a time to maintain order
            if block_id == pid:
                block_start = block_id * BLOCK_SIZE
                
                for i in range(BLOCK_SIZE):
                    idx = block_start + i
                    if idx < n_elements:
                        val = tl.load(b_ptr + idx)
                        if val > 0.0:
                            # Get current counter value and increment
                            current_count = tl.load(counter_ptr)
                            tl.store(a_ptr + current_count, val)
                            tl.store(counter_ptr, current_count + 1)
        
        # Synchronize between blocks (this is simplified - actual implementation needs barriers)
        tl.debug_barrier()

# Simplified single-threaded version that maintains correctness
@triton.jit 
def s341_kernel_sequential(b_ptr, a_ptr, counter_ptr, n_elements):
    # Single program processes all elements to maintain order
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first program runs
        j = 0
        for i in range(n_elements):
            val = tl.load(b_ptr + i)
            if val > 0.0:
                tl.store(a_ptr + j, val)
                j += 1
        
        # Store final count
        tl.store(counter_ptr, j)

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Clear output array
    a.fill_(0.0)
    
    # Counter for number of elements packed
    counter = torch.zeros(1, dtype=torch.int32, device=a.device)
    
    # Use single program to maintain sequential order
    grid = (1,)
    
    s341_kernel_sequential[grid](
        b, a, counter, n_elements
    )