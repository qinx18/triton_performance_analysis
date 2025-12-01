import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel_strip(
    flat_2d_array_ptr,
    a_ptr,
    strip_start,
    strip_size,
    vl: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s424 processing one strip.
    Each strip can be fully parallelized.

    For strip starting at strip_start:
      flat_2d_array[vl + strip_start + i + 1] = flat_2d_array[strip_start + i] + a[strip_start + i]
    where i ranges from 0 to strip_size-1
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid elements within this strip
    mask = offsets < strip_size

    # Calculate actual indices
    i = strip_start + offsets

    # Load input data: flat_2d_array[i] + a[i]
    flat_vals = tl.load(flat_2d_array_ptr + i, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + i, mask=mask, other=0.0)

    # Compute result
    result = flat_vals + a_vals

    # Store to flat_2d_array[vl + i + 1]
    output_offsets = vl + i + 1
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=mask)

def s424_triton(a, flat_2d_array):
    """
    Triton implementation of TSVC s424 - Multi-kernel version.
    Launches one kernel per strip.

    Original C code:
    int vl = 63;
    xx = flat_2d_array + vl;
    for (int i = 0; i < LEN_1D - 1; i++) {
        xx[i+1] = flat_2d_array[i] + a[i];
    }

    This is: flat_2d_array[64+i] = flat_2d_array[i] + a[i]

    The operation has overlapping reads/writes when i >= 64.
    Strip 0 (i=0..63): reads [0..63], writes [64..127] - independent
    Strip 1 (i=64..127): reads [64..127], writes [128..191] - depends on Strip 0
    Strip 2 (i=128..191): reads [128..191], writes [192..255] - depends on Strip 1

    Solution: Process in strips of 64, each strip fully parallel, but strips sequential.
    """
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()

    vl = 63
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1

    if n_elements <= 0:
        return flat_2d_array

    # Process in strips of 64 elements
    STRIP_SIZE = 64
    BLOCK_SIZE = 256

    # Process each strip sequentially (strips cannot be parallelized due to dependencies)
    num_strips = (n_elements + STRIP_SIZE - 1) // STRIP_SIZE

    for strip_idx in range(num_strips):
        strip_start = strip_idx * STRIP_SIZE
        strip_size = min(STRIP_SIZE, n_elements - strip_start)

        # Launch kernel for this strip (elements within strip can be parallel)
        grid_size = triton.cdiv(strip_size, BLOCK_SIZE)

        s424_kernel_strip[(grid_size,)](
            flat_2d_array,
            a,
            strip_start,
            strip_size,
            vl,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return flat_2d_array
