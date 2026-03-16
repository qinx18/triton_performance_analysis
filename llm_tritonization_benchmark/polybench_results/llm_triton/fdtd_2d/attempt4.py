import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_phase1_kernel(ey_ptr, _fict_ptr, t, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_block = tl.program_id(0)
    j_start = j_block * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < NY
    
    fict_val = tl.load(_fict_ptr + t)
    ey_offsets = j_offsets  # ey[0][j] = 0 * NY + j
    tl.store(ey_ptr + ey_offsets, fict_val, mask=j_mask)

@triton.jit
def fdtd_2d_phase2_kernel(ey_ptr, hz_ptr, i, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_block = tl.program_id(0)
    j_start = j_block * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < NY
    
    ey_offsets = i * NY + j_offsets
    hz_offsets = i * NY + j_offsets
    hz_prev_offsets = (i - 1) * NY + j_offsets
    
    ey_vals = tl.load(ey_ptr + ey_offsets, mask=j_mask)
    hz_vals = tl.load(hz_ptr + hz_offsets, mask=j_mask)
    hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets, mask=j_mask)
    
    new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
    tl.store(ey_ptr + ey_offsets, new_ey, mask=j_mask)

@triton.jit
def fdtd_2d_phase3_kernel(ex_ptr, hz_ptr, i, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_block = tl.program_id(0)
    j_start = j_block * BLOCK_SIZE + 1  # j starts from 1
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < NY
    
    ex_offsets = i * NY + j_offsets
    hz_offsets = i * NY + j_offsets
    hz_prev_offsets = i * NY + (j_offsets - 1)
    
    ex_vals = tl.load(ex_ptr + ex_offsets, mask=j_mask)
    hz_vals = tl.load(hz_ptr + hz_offsets, mask=j_mask)
    hz_prev_vals = tl.load(hz_ptr + hz_prev_offsets, mask=j_mask)
    
    new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
    tl.store(ex_ptr + ex_offsets, new_ex, mask=j_mask)

@triton.jit
def fdtd_2d_phase4_kernel(hz_ptr, ex_ptr, ey_ptr, i, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_block = tl.program_id(0)
    j_start = j_block * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < (NY - 1)
    
    hz_offsets = i * NY + j_offsets
    ex_next_offsets = i * NY + (j_offsets + 1)
    ex_curr_offsets = i * NY + j_offsets
    ey_next_offsets = (i + 1) * NY + j_offsets
    ey_curr_offsets = i * NY + j_offsets
    
    hz_vals = tl.load(hz_ptr + hz_offsets, mask=j_mask)
    ex_next_vals = tl.load(ex_ptr + ex_next_offsets, mask=j_mask)
    ex_curr_vals = tl.load(ex_ptr + ex_curr_offsets, mask=j_mask)
    ey_next_vals = tl.load(ey_ptr + ey_next_offsets, mask=j_mask)
    ey_curr_vals = tl.load(ey_ptr + ey_curr_offsets, mask=j_mask)
    
    new_hz = hz_vals - 0.7 * ((ex_next_vals - ex_curr_vals) + (ey_next_vals - ey_curr_vals))
    tl.store(hz_ptr + hz_offsets, new_hz, mask=j_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    grid1 = (triton.cdiv(NY, BLOCK_SIZE),)
    grid2 = (triton.cdiv(NY, BLOCK_SIZE),)
    grid3 = (triton.cdiv(NY - 1, BLOCK_SIZE),)
    grid4 = (triton.cdiv(NY - 1, BLOCK_SIZE),)
    
    for t in range(TMAX):
        # Phase 1: Set boundary conditions for ey[0][j]
        fdtd_2d_phase1_kernel[grid1](ey, _fict_, t, NY, BLOCK_SIZE)
        
        # Phase 2: Update ey[i][j] for i >= 1
        for i in range(1, NX):
            fdtd_2d_phase2_kernel[grid2](ey, hz, i, NY, BLOCK_SIZE)
        
        # Phase 3: Update ex[i][j] for j >= 1
        for i in range(NX):
            fdtd_2d_phase3_kernel[grid3](ex, hz, i, NY, BLOCK_SIZE)
        
        # Phase 4: Update hz[i][j] for i < NX-1, j < NY-1
        for i in range(NX - 1):
            fdtd_2d_phase4_kernel[grid4](hz, ex, ey, i, NY, BLOCK_SIZE)