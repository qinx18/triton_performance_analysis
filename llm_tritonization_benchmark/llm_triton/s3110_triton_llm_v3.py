import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, LEN_2D: tl.constexpr, output_ptr):
    # This is a global maximum reduction - each program handles a portion of the 2D array
    pid = tl.program_id(0)
    BLOCK_SIZE: tl.constexpr = 256
    
    # Initialize local max and indices
    local_max = float('-inf')
    local_xindex = 0
    local_yindex = 0
    
    # Each program processes a block of elements
    start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process elements in chunks
    for chunk_start in range(start_idx, LEN_2D * LEN_2D, tl.num_programs(0) * BLOCK_SIZE):
        current_offsets = chunk_start + offsets
        mask = current_offsets < LEN_2D * LEN_2D
        
        # Load values
        vals = tl.load(aa_ptr + current_offsets, mask=mask, other=float('-inf'))
        
        # Find max in this chunk
        for i in range(BLOCK_SIZE):
            if (chunk_start + i) < LEN_2D * LEN_2D:
                flat_idx = chunk_start + i
                val = tl.load(aa_ptr + flat_idx)
                if val > local_max:
                    local_max = val
                    local_xindex = flat_idx // LEN_2D
                    local_yindex = flat_idx % LEN_2D
    
    # Store results (max, xindex, yindex) for this program
    tl.store(output_ptr + pid * 3, local_max)
    tl.store(output_ptr + pid * 3 + 1, local_xindex)
    tl.store(output_ptr + pid * 3 + 2, local_yindex)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    num_programs = min(128, (LEN_2D * LEN_2D + 255) // 256)
    
    # Output buffer for partial results: [max, xindex, yindex] for each program
    partial_results = torch.empty(num_programs * 3, dtype=aa.dtype, device=aa.device)
    
    # Launch kernel
    s3110_kernel[(num_programs,)](
        aa.flatten(),
        LEN_2D,
        partial_results
    )
    
    # Reduce partial results on CPU to find global maximum
    partial_results_cpu = partial_results.cpu().numpy()
    
    global_max = float('-inf')
    global_xindex = 0
    global_yindex = 0
    
    for i in range(num_programs):
        max_val = partial_results_cpu[i * 3]
        xindex = int(partial_results_cpu[i * 3 + 1])
        yindex = int(partial_results_cpu[i * 3 + 2])
        
        if max_val > global_max:
            global_max = max_val
            global_xindex = xindex
            global_yindex = yindex
    
    chksum = global_max + float(global_xindex) + float(global_yindex)
    
    return global_max + global_xindex + 1 + global_yindex + 1