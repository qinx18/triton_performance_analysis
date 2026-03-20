import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_step1_kernel(
    _fict_ptr, ey_ptr, t: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NY
    
    fict_val = tl.load(_fict_ptr + t)
    ey_indices = offsets
    tl.store(ey_ptr + ey_indices, fict_val, mask=mask)

@triton.jit
def fdtd_2d_step2_kernel(
    ey_ptr, hz_ptr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid + 1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_start = 0
    
    for j_block in range(0, NY, BLOCK_SIZE):
        j_offsets = j_block + offsets
        mask = j_offsets < NY
        
        ey_indices = i * NY + j_offsets
        hz_indices = i * NY + j_offsets
        hz_prev_indices = (i - 1) * NY + j_offsets
        
        ey_vals = tl.load(ey_ptr + ey_indices, mask=mask)
        hz_vals = tl.load(hz_ptr + hz_indices, mask=mask)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=mask)
        
        new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ey_ptr + ey_indices, new_ey, mask=mask)

@triton.jit
def fdtd_2d_step3_kernel(
    ex_ptr, hz_ptr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(1, NY, BLOCK_SIZE):
        j_offsets = j_block + offsets
        mask = j_offsets < NY
        
        ex_indices = i * NY + j_offsets
        hz_indices = i * NY + j_offsets
        hz_prev_indices = i * NY + (j_offsets - 1)
        
        ex_vals = tl.load(ex_ptr + ex_indices, mask=mask)
        hz_vals = tl.load(hz_ptr + hz_indices, mask=mask)
        hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=mask)
        
        new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
        tl.store(ex_ptr + ex_indices, new_ex, mask=mask)

@triton.jit
def fdtd_2d_step4_kernel(
    ex_ptr, ey_ptr, hz_ptr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_block in range(0, NY - 1, BLOCK_SIZE):
        j_offsets = j_block + offsets
        mask = j_offsets < (NY - 1)
        
        hz_indices = i * NY + j_offsets
        ex_indices = i * NY + j_offsets
        ex_next_indices = i * NY + (j_offsets + 1)
        ey_indices = i * NY + j_offsets
        ey_next_indices = (i + 1) * NY + j_offsets
        
        hz_vals = tl.load(hz_ptr + hz_indices, mask=mask)
        ex_vals = tl.load(ex_ptr + ex_indices, mask=mask)
        ex_next_vals = tl.load(ex_ptr + ex_next_indices, mask=mask)
        ey_vals = tl.load(ey_ptr + ey_indices, mask=mask)
        ey_next_vals = tl.load(ey_ptr + ey_next_indices, mask=mask)
        
        new_hz = hz_vals - 0.7 * (ex_next_vals - ex_vals + ey_next_vals - ey_vals)
        tl.store(hz_ptr + hz_indices, new_hz, mask=mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 64
    
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t]
        grid1 = (triton.cdiv(NY, BLOCK_SIZE),)
        fdtd_2d_step1_kernel[grid1](_fict_, ey, t, NY, BLOCK_SIZE)
        
        # Step 2: ey[i][j] update for i=1 to NX-1
        if NX > 1:
            grid2 = (NX - 1,)
            fdtd_2d_step2_kernel[grid2](ey, hz, NY, BLOCK_SIZE)
        
        # Step 3: ex[i][j] update
        grid3 = (NX,)
        fdtd_2d_step3_kernel[grid3](ex, hz, NY, BLOCK_SIZE)
        
        # Step 4: hz[i][j] update
        if NX > 1:
            grid4 = (NX - 1,)
            fdtd_2d_step4_kernel[grid4](ex, ey, hz, NY, BLOCK_SIZE)