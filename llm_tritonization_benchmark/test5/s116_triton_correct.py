import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s116 - processes 5 consecutive elements with dependencies.
    Each program processes groups of 5 elements (i=0,5,10,15,...).

    FIXED: Treats BLOCK_SIZE as number of GROUPS, not element indices.
    """
    # Get program ID - this is the GROUP id, not element id
    pid = tl.program_id(axis=0)

    # Calculate which groups this block handles
    # Each group processes 5 consecutive elements starting at indices 0,5,10,15,...
    group_ids = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Convert group IDs to actual array indices (groups start at 0, 5, 10, ...)
    base_offsets = group_ids * 5

    # Mask for valid groups - ensure we don't go beyond n_elements - 5
    mask = base_offsets < (n_elements - 5)

    # Load 6 consecutive values starting from each base offset
    # We need a[i], a[i+1], ..., a[i+5] for each group
    a0 = tl.load(a_ptr + base_offsets, mask=mask, other=0.0)
    a1 = tl.load(a_ptr + base_offsets + 1, mask=mask, other=0.0)
    a2 = tl.load(a_ptr + base_offsets + 2, mask=mask, other=0.0)
    a3 = tl.load(a_ptr + base_offsets + 3, mask=mask, other=0.0)
    a4 = tl.load(a_ptr + base_offsets + 4, mask=mask, other=0.0)
    a5 = tl.load(a_ptr + base_offsets + 5, mask=mask, other=0.0)

    # Compute the 5 operations with proper dependencies
    # a[i] = a[i + 1] * a[i]
    # a[i + 1] = a[i + 2] * a[i + 1]
    # a[i + 2] = a[i + 3] * a[i + 2]
    # a[i + 3] = a[i + 4] * a[i + 3]
    # a[i + 4] = a[i + 5] * a[i + 4]
    new_a0 = a1 * a0
    new_a1 = a2 * a1
    new_a2 = a3 * a2
    new_a3 = a4 * a3
    new_a4 = a5 * a4

    # Store results back to memory
    tl.store(a_ptr + base_offsets, new_a0, mask=mask)
    tl.store(a_ptr + base_offsets + 1, new_a1, mask=mask)
    tl.store(a_ptr + base_offsets + 2, new_a2, mask=mask)
    tl.store(a_ptr + base_offsets + 3, new_a3, mask=mask)
    tl.store(a_ptr + base_offsets + 4, new_a4, mask=mask)


def s116_triton(a):
    """
    Triton implementation of TSVC s116 - linear dependence testing with unrolling.

    Original C code:
    for (int i = 0; i < LEN_1D - 5; i += 5) {
        a[i] = a[i + 1] * a[i];
        a[i + 1] = a[i + 2] * a[i + 1];
        a[i + 2] = a[i + 3] * a[i + 2];
        a[i + 3] = a[i + 4] * a[i + 3];
        a[i + 4] = a[i + 5] * a[i + 4];
    }

    Args:
        a: Input/output tensor

    Returns:
        torch.Tensor: Modified array a
    """
    a = a.contiguous()
    n = a.size(0)

    if n <= 5:
        return a

    # Calculate number of groups of 5 elements we can process
    # Each group starts at i = 0, 5, 10, 15, ...
    n_groups = (n - 5) // 5
    if n_groups == 0:
        return a

    # BLOCK_SIZE is the number of GROUPS each thread block processes
    BLOCK_SIZE = 64  # Process 64 groups per block

    # Calculate grid size - number of thread blocks needed
    grid_size = triton.cdiv(n_groups, BLOCK_SIZE)

    # Launch kernel with appropriate grid
    s116_kernel[(grid_size,)](
        a,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return a
