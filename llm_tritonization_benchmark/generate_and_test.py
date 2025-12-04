#!/usr/bin/env python3
"""
Integrated Generation and Testing Pipeline for TSVC Functions

For every generated function:
1. Generate Triton implementation using Claude API
2. Test immediately after generation
3. If correctness test passed → move to next function
4. If numerical error → retry with "numerical error" + "last attempt" + original prompt
5. If non-numerical error → retry with exact error description + "last attempt" + original prompt
6. Maximum 3 attempts per function

Usage:
    python generate_and_test.py              # Process all functions
    python generate_and_test.py s271 s241    # Process specific functions
"""

import os
import sys
import subprocess
import anthropic
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Import the extracted function database
sys.path.append(str(Path(__file__).parent / "utilities"))
from tsvc_functions_db import TSVC_FUNCTIONS

# Add PET analysis directory to path
sys.path.insert(0, "/home/qinxiao/workspace/pet/isl_analysis")
try:
    from compute_war_dependences import analyze_kernel_war
    HAS_WAR_ANALYSIS = True
except ImportError:
    HAS_WAR_ANALYSIS = False
    analyze_kernel_war = None

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None

# Paths
TSVC_SOURCE = "/home/qinxiao/workspace/TSVC_2/src/archive/tsvc_orig.c"
KERNELS_DIR = "/home/qinxiao/workspace/pet/isl_analysis/kernels"
PARALLEL_DIMS_ANALYSIS_FILE = "/home/qinxiao/workspace/pet/isl_analysis/results/parallel_dims_analysis.txt"
BASELINES_DIR = Path("baselines")

MAX_ATTEMPTS = 3

# Helper functions defined in TSVC
HELPER_FUNCTIONS = {
    "test": {
        "code": """real_t test(real_t* A){
  real_t s = (real_t)0.0;
  for (int i = 0; i < 4; i++)
    s += A[i];
  return s;
}""",
        "description": "Sums the first 4 elements of array A"
    },
    "f": {
        "code": """real_t f(real_t a, real_t b){
    return a*b;
}""",
        "description": "Returns the product of a and b (a*b)"
    }
}


# ============================================================================
# From generate_llm_triton.py - Code generation functions
# ============================================================================

def find_used_helper_functions(func_body: str) -> List[str]:
    """Find which helper functions are called in the function body."""
    used = []
    for helper_name in HELPER_FUNCTIONS:
        if re.search(rf'\b{helper_name}\s*\(', func_body):
            used.append(helper_name)
    return used


def extract_tsvc_function(func_name: str) -> Optional[dict]:
    """Extract a function from the original TSVC source file."""
    with open(TSVC_SOURCE, 'r') as f:
        content = f.read()

    pattern = rf'real_t {func_name}\s*\(struct args_t \* func_args\)\s*\{{'
    match = re.search(pattern, content)
    if not match:
        return None

    start = match.start()
    brace_count = 0
    in_func = False
    end = start

    for i, char in enumerate(content[start:], start):
        if char == '{':
            brace_count += 1
            in_func = True
        elif char == '}':
            brace_count -= 1
            if in_func and brace_count == 0:
                end = i + 1
                break

    func_body = content[start:end]
    local_vars = extract_local_variables(func_body)
    kernel_loop = extract_kernel_loop(func_body)

    desc_match = re.search(rf'//.*?\n.*?real_t {func_name}', content[max(0, start - 200):start])
    description = ""
    if desc_match:
        desc_lines = desc_match.group(0).split('\n')
        description = '\n'.join(line for line in desc_lines if line.strip().startswith('//'))

    helper_functions_used = find_used_helper_functions(func_body)

    return {
        'name': func_name,
        'full_body': func_body,
        'local_vars': local_vars,
        'kernel_loop': kernel_loop,
        'description': description,
        'helper_functions': helper_functions_used
    }


def extract_local_variables(func_body: str) -> List[str]:
    """Extract local variable declarations from function body."""
    variables = []
    var_patterns = [
        r'^\s*(int\s+(?!nl\b)(?!i\b)(?!j\b)(?!k\b)\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(real_t\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(float\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(double\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
    ]

    lines = func_body.split('\n')
    for line in lines:
        if 'for' in line and ('nl' in line or 'i =' in line or 'j =' in line):
            continue
        if 'gettimeofday' in line:
            continue

        for pattern in var_patterns:
            match = re.match(pattern, line)
            if match:
                decl = match.group(1).strip()
                variables.append(decl)
                break

    return variables


def extract_kernel_loop(func_body: str) -> str:
    """Extract the kernel loop from function body (excluding outer nl loop)."""
    lines = func_body.split('\n')
    loop_lines = []
    in_main_loop = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        if 'initialise_arrays' in stripped or 'gettimeofday' in stripped:
            continue
        if 'dummy(' in stripped or 'calc_checksum' in stripped:
            continue
        if stripped.startswith('//'):
            continue

        if 'for' in stripped and 'nl' in stripped:
            in_main_loop = True
            if '{' in stripped:
                brace_depth = 1
                remainder = stripped[stripped.find('{') + 1:].strip()
                if remainder:
                    loop_lines.append(remainder)
            continue

        if in_main_loop:
            for char in stripped:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

            if brace_depth > 0:
                if 'dummy(' not in stripped:
                    loop_lines.append(line)
            else:
                break

    return '\n'.join(loop_lines)


def get_exact_function_signature(kernel_name: str) -> Optional[str]:
    """Build the exact function signature from the TSVC database."""
    if kernel_name not in TSVC_FUNCTIONS:
        return None

    func_info = TSVC_FUNCTIONS[kernel_name]
    params = []

    if 'arrays' in func_info:
        arrays = sorted(func_info['arrays'].keys())
        params.extend(arrays)

    if 'scalar_params' in func_info:
        scalars = [p for p in sorted(func_info['scalar_params'].keys()) if p != 'iterations']
        params.extend(scalars)

    return ", ".join(params) if params else None


def load_parallelization_analysis(kernel_name: str) -> Optional[dict]:
    """Load parallelization analysis for a kernel."""
    if not os.path.exists(PARALLEL_DIMS_ANALYSIS_FILE):
        return None

    with open(PARALLEL_DIMS_ANALYSIS_FILE, 'r') as f:
        content = f.read()

    eq40 = '=' * 40
    eq80 = '=' * 80
    pattern = rf'{re.escape(eq40)}\nKernel: {re.escape(kernel_name)}\n{re.escape(eq40)}\n(.*?)(?=\n{re.escape(eq40)}\nKernel:|\n\n{re.escape(eq80)}|$)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    section = match.group(1).strip()
    result = {
        'kernel': kernel_name,
        'c_code': '',
        'dims': [],
        'is_triangular': False,
        'triangular_info': None,
        'self_dependencies': [],
        'options': [],
        'summary': ''
    }

    c_match = re.search(r'C Code: (.*?)(?=\n\nStatement|\n\n|$)', section, re.DOTALL)
    if c_match:
        result['c_code'] = c_match.group(1).strip()

    dims_match = re.search(r'Dimensions: \[(.*?)\]', section)
    if dims_match:
        result['dims'] = [d.strip().strip("'") for d in dims_match.group(1).split(',')]

    tri_match = re.search(r'Triangular bounds: (\w+) < (\w+)', section)
    if tri_match:
        result['is_triangular'] = True
        result['triangular_info'] = {'smaller': tri_match.group(1), 'larger': tri_match.group(2)}

    deps_match = re.findall(r'- (\w+): write \[(.*?)\], read \[(.*?)\]', section)
    for arr, write_expr, read_expr in deps_match:
        result['self_dependencies'].append({
            'array': arr,
            'write_expr': write_expr,
            'read_expr': read_expr
        })

    opts_section = re.search(r'Parallelization Options:(.*?)(?=\n\n  SUMMARY|\n\n\n|\n\n$)', section, re.DOTALL)
    if opts_section:
        opt_lines = opts_section.group(1).split('\n')
        current_opt = None

        for line in opt_lines:
            opt_match = re.match(r'\s*(\w+)-sequential,\s*(\w+)-parallel:\s*(VALID|INVALID)\s*\((\w+)\)', line)
            if opt_match:
                current_opt = {
                    'sequential_dim': opt_match.group(1),
                    'parallel_dim': opt_match.group(2),
                    'valid': opt_match.group(3) == 'VALID',
                    'parallelism_type': opt_match.group(4),
                    'triton_strategy': None,
                    'issues': [],
                    'explanations': []
                }
                result['options'].append(current_opt)
            elif current_opt and 'Triton Strategy:' in line:
                strat_match = re.search(r'Triton Strategy:\s*(\S+)', line)
                if strat_match:
                    current_opt['triton_strategy'] = strat_match.group(1)

    summary_match = re.search(r'SUMMARY: (.+)', section)
    if summary_match:
        result['summary'] = summary_match.group(1).strip()

    return result


def load_war_analysis(kernel_name: str) -> Optional[dict]:
    """Load WAR anti-dependency analysis for a kernel."""
    if not HAS_WAR_ANALYSIS or analyze_kernel_war is None:
        return None

    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    try:
        return analyze_kernel_war(kernel_file)
    except Exception:
        return None


def build_war_instructions(kernel_name: str, war_result: dict) -> str:
    """Build specific instructions for handling WAR dependencies."""
    if not war_result or war_result['parallelization_safe']:
        return ""

    arrays_to_copy = war_result['arrays_needing_copy']

    instructions = f"""
## CRITICAL: WAR Race Condition Handling Required

This kernel has WAR (Write-After-Read) anti-dependencies that cause race conditions in parallel execution.
**Arrays requiring read-only copy**: {arrays_to_copy}

### Required Solution: Read-Only Copy Pattern
Pass a **read-only copy** of the array to the kernel. All threads load from the copy (immutable)
and store results to the original array.

### Implementation Template
```python
# In wrapper function - create read-only copy BEFORE launching kernel:
def {kernel_name}_triton(...):
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_copy = {arr}.clone()  # Read-only copy\n"

    instructions += f"""
    # Pass BOTH original (for writes) AND copy (for reads) to kernel
    {kernel_name}_kernel[grid](
"""
    for arr in arrays_to_copy:
        instructions += f"        {arr},        # Write destination (original)\n"
        instructions += f"        {arr}_copy,   # Read source (immutable copy)\n"

    instructions += """        ...other_args...
    )
```

**CRITICAL: Use forward iteration** - ALWAYS use forward iteration with ascending offsets.
"""
    return instructions


def build_parallelization_instructions(kernel_name: str, analysis: Optional[dict]) -> str:
    """Build specific instructions for parallelization strategy."""
    if not analysis:
        return ""

    valid_options = [opt for opt in analysis['options'] if opt['valid']]
    invalid_options = [opt for opt in analysis['options'] if not opt['valid']]

    if not valid_options and not invalid_options:
        return ""

    lines = []
    lines.append("")
    lines.append("## CRITICAL: Parallelization Strategy")
    lines.append("")
    lines.append(f"**Loop structure**: `{analysis['c_code']}`")
    lines.append(f"**Dimensions**: {analysis['dims']}")
    if analysis['is_triangular']:
        tri = analysis['triangular_info']
        lines.append(f"**Triangular bounds**: {tri['smaller']} < {tri['larger']}")
    lines.append("")

    if analysis['self_dependencies']:
        lines.append("**Data dependencies detected**:")
        for dep in analysis['self_dependencies']:
            lines.append(f"- Array `{dep['array']}`: write to `[{dep['write_expr']}]`, read from `[{dep['read_expr']}]`")
        lines.append("")

    if len(valid_options) >= 1:
        opt = valid_options[0]
        seq_dim = opt['sequential_dim']
        par_dim = opt['parallel_dim']
        par_type = opt['parallelism_type']
        triton_strategy = opt.get('triton_strategy', 'MULTI_KERNEL_LAUNCH')

        lines.append(f"### Required Strategy: {seq_dim}-sequential, {par_dim}-parallel")
        lines.append("")

        if triton_strategy == 'SINGLE_KERNEL_INLOOP':
            # Prefer in-kernel loop - more efficient, single kernel launch
            lines.append("**Implementation Pattern (SINGLE KERNEL with in-kernel loop):**")
            lines.append(f"- Python wrapper: launch ONE kernel with `grid = (triton.cdiv({par_dim}_size, BLOCK_SIZE),)`")
            lines.append(f"- Triton kernel: use `for {seq_dim} in range(...)` loop INSIDE the kernel")
            lines.append(f"- Triton kernel: parallelize `{par_dim}` values using VECTORIZED operations within each block")
            lines.append("")
            lines.append("**Why in-kernel loop is safe:**")
            lines.append(f"- The sequential dimension `{seq_dim}` has dependencies that require ordering")
            lines.append(f"- But within each `{seq_dim}` iteration, all `{par_dim}` values are independent")
            lines.append(f"- Reading from previous `{seq_dim}` iteration is safe because the in-kernel loop ensures ordering")
            lines.append("")
            lines.append("**Example structure:**")
            lines.append("```python")
            lines.append("@triton.jit")
            lines.append(f"def kernel(...):")
            lines.append(f"    {par_dim}_offsets = tl.arange(0, BLOCK_SIZE)")
            lines.append(f"    {par_dim}_idx = pid * BLOCK_SIZE + {par_dim}_offsets")
            lines.append(f"    for {seq_dim} in range(start, end):  # Sequential loop INSIDE kernel")
            lines.append(f"        # Load, compute, store for this {seq_dim} iteration")
            lines.append("        ...")
            lines.append("")
            lines.append(f"# Wrapper: single kernel launch!")
            lines.append(f"grid = (triton.cdiv({par_dim}_size, BLOCK_SIZE),)")
            lines.append("kernel[grid](...)")
            lines.append("```")
            lines.append("")
        else:
            # Fallback to sequential kernel launches
            lines.append("**Implementation Pattern (Sequential kernel launches):**")
            lines.append(f"- Python wrapper: loop over `{seq_dim}` sequentially, launching kernel each iteration")
            lines.append(f"- Triton kernel: parallelize ALL `{par_dim}` values using VECTORIZED operations")
            lines.append("")

        if par_type == 'reduction':
            lines.append(f"**Reduction pattern:** Use `tl.sum()` to reduce across {par_dim} dimension")
            lines.append("")

    if invalid_options:
        lines.append("### INVALID Parallelization (DO NOT USE)")
        for opt in invalid_options:
            lines.append(f"**{opt['sequential_dim']}-sequential, {opt['parallel_dim']}-parallel**: INCORRECT - causes race conditions")
        lines.append("")

    return "\n".join(lines)


def build_base_prompt(kernel_name: str, tsvc_func: dict, exact_sig: str,
                      war_section: str, par_section: str) -> str:
    """Build the base prompt for Triton generation."""
    c_code_section = tsvc_func['kernel_loop']
    if tsvc_func['local_vars']:
        local_vars_str = '\n'.join(f"    {v};" for v in tsvc_func['local_vars'])
        c_code_section = f"// Local variables:\n{local_vars_str}\n\n// Kernel loop:\n{c_code_section}"

    helper_section = ""
    if tsvc_func['helper_functions']:
        helper_funcs_code = []
        for helper_name in tsvc_func['helper_functions']:
            helper_info = HELPER_FUNCTIONS[helper_name]
            helper_funcs_code.append(f"// {helper_info['description']}\n{helper_info['code']}")
        helper_section = f"""

## Helper Functions Called in This Code:
```c
{chr(10).join(helper_funcs_code)}
```
"""

    prompt = f"""I have an original TSVC (Test Suite for Vectorizing Compilers) C function that I want to implement in Triton for GPU acceleration.

## Original TSVC C Code:
```c
{tsvc_func['description']}
{tsvc_func['full_body']}
```

## Kernel Loop to Implement:
```c
{c_code_section}
```
{helper_section}{war_section}{par_section}

## Array Information:
- Arrays `a`, `b`, `c`, `d`, `e` are 1D float arrays of size LEN_1D (typically 32000)
- Arrays `aa`, `bb`, `cc`, `tt` are 2D float arrays of size LEN_2D x LEN_2D (typically 256x256)
- `flat_2d_array` is a 1D float array of size LEN_2D*LEN_2D
- `indx` is a 1D int array of size LEN_1D

## Requirements:
Please generate a complete Triton implementation that:
1. Includes a @triton.jit kernel function named `{kernel_name}_kernel`
2. Includes a Python wrapper function named `{kernel_name}_triton`
3. The wrapper should accept ONLY the PyTorch tensor arrays used in the computation
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the C code (same computation, same results)

## CRITICAL: Function Signature Requirements
**DO NOT include** the `iterations` parameter or the outer `for (int nl = ...)` timing loop.

**REQUIRED function signature (use EXACTLY these parameter names):**
```python
def {kernel_name}_triton({exact_sig if exact_sig else ''}):
    ...  # Just the kernel computation, NO timing loop
```

IMPORTANT:
- Use EXACTLY the parameter names shown in the required function signature above
- Do NOT implement the outer timing loop
- If WAR dependencies are shown above, you MUST use the read-only copy pattern
- If parallelization analysis is shown above, you MUST follow the specified parallelization strategy

## CRITICAL: Triton Compilation Rules

**NEVER use `tl.arange()` inside a for loop:**
```python
# WRONG
for block_start in range(0, n, BLOCK_SIZE):
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # ERROR!

# CORRECT
offsets = tl.arange(0, BLOCK_SIZE)  # Define once at start
for block_start in range(0, n, BLOCK_SIZE):
    current_offsets = block_start + offsets
```

**NEVER use scalar indexing inside @triton.jit kernel:**
```python
# WRONG
for i in range(BLOCK_SIZE):
    val = tensor[i]

# CORRECT
mask = offsets < n_elements
vals = tl.load(ptr + offsets, mask=mask)
```

**NEVER use non-existent Triton functions:**
```python
# WRONG - tl.mul, tl.div, tl.add, tl.any, tl.cdiv don't exist
# CORRECT - use Python operators: a * b, a / b, a + b
# For cdiv, use triton.cdiv() in wrapper only
```

**NEVER use Python lists inside @triton.jit kernels**

**NEVER use break/continue inside @triton.jit kernels**

**Pass tensors directly to kernels, NOT data_ptr()**

**NEVER use chained comparisons (use separate comparisons with &)**

**NEVER load from bare pointer without offset vector**

Provide ONLY the Python code, no additional explanation."""

    return prompt


def generate_triton_with_retry(kernel_name: str, original_prompt: str,
                               last_attempt: str, error_info: dict,
                               attempt_num: int) -> Tuple[str, str]:
    """Generate Triton code with error feedback for retry."""

    if error_info['type'] == 'numerical':
        error_section = f"""
## PREVIOUS ATTEMPT FAILED - NUMERICAL ERROR

Your last attempt produced incorrect numerical results. The output values don't match the expected PyTorch baseline.

**Error type**: Numerical error (values don't match)
**Max error observed**: {error_info.get('max_error', 'unknown')}

Please fix the numerical computation. Common causes:
- Incorrect loop bounds or indices
- Wrong array access patterns
- Missing or incorrect operations
- Off-by-one errors in indexing

## LAST ATTEMPT (DO NOT REPEAT THE SAME MISTAKES):
```python
{last_attempt}
```

"""
    else:
        error_section = f"""
## PREVIOUS ATTEMPT FAILED - NON-NUMERICAL ERROR

Your last attempt failed with a compilation or runtime error:

**Error type**: {error_info['type']}
**Error message**:
```
{error_info['message']}
```

Please fix the error. Make sure to:
- Follow all Triton compilation rules
- Use correct Triton API functions
- Avoid Python constructs not supported in Triton kernels

## LAST ATTEMPT (DO NOT REPEAT THE SAME MISTAKES):
```python
{last_attempt}
```

"""

    retry_prompt = f"""{error_section}

## ORIGINAL TASK:
{original_prompt}

This is attempt {attempt_num} of {MAX_ATTEMPTS}. Please provide a corrected implementation.
Provide ONLY the Python code, no additional explanation."""

    print(f"    Retrying with error feedback (attempt {attempt_num}/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": retry_prompt}]
    )

    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, retry_prompt


def generate_triton_initial(kernel_name: str) -> Tuple[str, str, str]:
    """Generate initial Triton implementation. Returns (code, prompt, full_response)."""
    tsvc_func = extract_tsvc_function(kernel_name)
    if not tsvc_func:
        raise ValueError(f"Could not find function {kernel_name} in TSVC source")

    exact_sig = get_exact_function_signature(kernel_name)
    war_result = load_war_analysis(kernel_name)
    war_section = build_war_instructions(kernel_name, war_result)
    par_analysis = load_parallelization_analysis(kernel_name)
    par_section = build_parallelization_instructions(kernel_name, par_analysis)

    prompt = build_base_prompt(kernel_name, tsvc_func, exact_sig, war_section, par_section)

    print(f"  Generating Triton code (attempt 1/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_response = f"""# LLM-Generated Triton Implementation for {kernel_name}
# Generated: {timestamp}
# Model: claude-sonnet-4-20250514

{'=' * 80}
PROMPT:
{'=' * 80}
{prompt}

{'=' * 80}
RESPONSE:
{'=' * 80}
{message.content[0].text}
"""

    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, prompt, full_response


# ============================================================================
# From auto_test_all_tsvc.py - Testing functions
# ============================================================================

def generate_pytorch_baseline(func_name: str, func_spec: dict) -> Optional[str]:
    """Generate PyTorch baseline implementation using LLM."""
    if not client:
        print(f"  No API key, skipping baseline generation for {func_name}")
        return None

    loop_code = func_spec['loop_code']
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})

    array_desc = ', '.join([f"{arr} ({mode})" for arr, mode in sorted(arrays.items())])
    filtered_scalars = {k: v for k, v in scalar_params.items() if k != 'iterations'}

    scalar_desc = ""
    if filtered_scalars:
        scalar_list = ', '.join(sorted(filtered_scalars.keys()))
        scalar_desc = f"\nScalar parameters (computational only): {scalar_list}"

    prompt = f"""Generate a PyTorch baseline implementation for TSVC function {func_name}.

Original C loop code:
```c
{loop_code}
```

Arrays used: {array_desc}
(r = read only, w = write only, rw = read-write){scalar_desc}

Generate a complete, simple, and correct PyTorch implementation:

1. Create a Python file with:
   - Start with: `import torch` at the very top
   - Function named `{func_name}_pytorch` that takes torch.Tensor arguments for ALL arrays{", plus computational scalar parameters" if filtered_scalars else ""}
   - Parameters should be in alphabetical order

2. Implementation guidelines:
   - Remove the outer timing loop completely
   - DO NOT include 'iterations' parameter
   - Use torch operations (torch.where for conditionals)
   - CRITICAL: Modify arrays IN-PLACE using slice assignment (a[:] = ...)
   - Do NOT return the arrays

Provide ONLY the complete Python code, no explanation."""

    print(f"  Generating PyTorch baseline...")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        code = message.content[0].text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        print(f"  Error generating baseline: {e}")
        return None


def generate_correctness_test(func_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate correctness test script for a specific attempt."""
    arrays = func_spec['arrays']
    has_offset = func_spec['has_offset']
    has_2d = func_spec.get('has_2d_arrays', False)
    scalar_params = func_spec.get('scalar_params', {})

    if has_offset:
        size_expr = "N + 10"
    else:
        size_expr = "N"

    array_inits = []
    for arr, mode in sorted(arrays.items()):
        if mode in ['r', 'rw', 'w']:
            if arr == 'ip':
                array_inits.append(f"            {arr} = torch.randint(0, {size_expr}, ({size_expr},), device='cuda', dtype=torch.long)")
            elif has_2d and len(arr) == 2 and arr[0] == arr[1]:
                array_inits.append(f"            {arr} = torch.randn({size_expr}, {size_expr}, device='cuda', dtype=torch.float32)")
            elif arr == 'flat_2d_array':
                if has_offset:
                    array_inits.append(f"            {arr} = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)")
                else:
                    array_inits.append(f"            {arr} = torch.randn(N * N, device='cuda', dtype=torch.float32)")
            else:
                array_inits.append(f"            {arr} = torch.randn({size_expr}, device='cuda', dtype=torch.float32)")

    for scalar_name in sorted(scalar_params.keys()):
        if scalar_name == 'k':
            array_inits.append(f"            {scalar_name} = 5")
        elif scalar_name == 't':
            array_inits.append(f"            {scalar_name} = 0.5")
        elif scalar_name in ['n1', 'n3']:
            if scalar_name == 'n1':
                array_inits.append(f"            {scalar_name} = 10")
            elif scalar_name == 'n3':
                array_inits.append(f"            {scalar_name} = 3")
        else:
            array_inits.append(f"            {scalar_name} = 1")

    array_init_str = '\n'.join(array_inits) if array_inits else "            pass"

    array_names = [arr for arr, mode in sorted(arrays.items()) if mode in ['r', 'rw', 'w']]
    output_arrays = [arr for arr, mode in sorted(arrays.items()) if mode in ['rw', 'w']]

    if not output_arrays and array_names:
        output_arrays = [array_names[0]]

    all_scalar_names = sorted(scalar_params.keys())

    pytorch_clones = []
    for arr in array_names:
        pytorch_clones.append(f"            {arr}_pt = {arr}.clone()")
    pytorch_clone_str = '\n'.join(pytorch_clones) if pytorch_clones else "            pass"

    triton_clones = []
    for arr in array_names:
        triton_clones.append(f"            {arr}_tr = {arr}.clone()")
    triton_clone_str = '\n'.join(triton_clones) if triton_clones else "            pass"

    if len(output_arrays) == 1:
        compare_str = f"            max_error = torch.max(torch.abs({output_arrays[0]}_pt - {output_arrays[0]}_tr)).item()"
        passed_check_str = f"passed = max_error < 1e-3 or torch.allclose({output_arrays[0]}_pt, {output_arrays[0]}_tr, rtol=1e-3, atol=1e-3)"
    elif len(output_arrays) > 1:
        compare_parts = [f"torch.max(torch.abs({arr}_pt - {arr}_tr)).item()" for arr in output_arrays]
        compare_str = f"            max_error = max([{', '.join(compare_parts)}])"
        passed_check_str = f"passed = max_error < 1e-3 or torch.allclose({output_arrays[0]}_pt, {output_arrays[0]}_tr, rtol=1e-3, atol=1e-3)"
    else:
        compare_str = "            max_error = 0.0"
        passed_check_str = "passed = True"

    available_arrays = array_names
    available_scalars = all_scalar_names

    if has_2d:
        test_sizes_str = "[64, 128, 256]"
    else:
        test_sizes_str = "[100, 1000, 10000]"

    test_code = f'''#!/usr/bin/env python3
"""
Correctness Test for {func_name}
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.{func_name}_baseline import {func_name}_pytorch
    from test10.llm_triton.{func_name}.attempt{attempt} import {func_name}_triton
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def test_correctness():
    test_sizes = {test_sizes_str}
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: {func_name}")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={{N:>6}}...", end=" ")

        try:
{array_init_str}

{pytorch_clone_str}

{triton_clone_str}

            pt_tensors = {{{', '.join([f'"{arr}": {arr}_pt' for arr in available_arrays])}}}
            tr_tensors = {{{', '.join([f'"{arr}": {arr}_tr' for arr in available_arrays])}}}
            scalars = {{{', '.join([f'"{s}": {s}' for s in available_scalars])}}}

            pt_args = build_args({func_name}_pytorch, pt_tensors, scalars)
            tr_args = build_args({func_name}_triton, tr_tensors, scalars)

            pytorch_result = {func_name}_pytorch(*pt_args)
            {func_name}_triton(*tr_args)

{compare_str}

            {passed_check_str}
            if passed:
                print(f"PASS  (max_err={{max_error:.2e}})")
            else:
                print(f"FAIL  (max_error={{max_error:.2e}})")
                all_passed = False

        except Exception as e:
            print(f"ERROR: {{e}}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("="*70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
'''

    return test_code


def run_test(func_name: str, test_file: Path) -> Tuple[bool, dict]:
    """Run the correctness test and parse results.

    Returns:
        (passed: bool, error_info: dict)
        error_info contains:
            - type: 'numerical', 'import', 'runtime', 'compilation', 'timeout'
            - message: error description
            - max_error: (for numerical errors) the max error value
    """
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd()
        )

        stdout = result.stdout
        stderr = result.stderr

        # Check for success
        if result.returncode == 0 and "All tests PASSED!" in stdout:
            return True, {}

        # Parse error type
        combined_output = stdout + stderr

        # Import error
        if "Import error:" in combined_output or "ImportError" in combined_output:
            return False, {
                'type': 'import',
                'message': combined_output[-2000:]  # Last 2000 chars
            }

        # Compilation error (Triton)
        if "CompilationError" in combined_output or "triton.compiler" in combined_output:
            return False, {
                'type': 'compilation',
                'message': combined_output[-2000:]
            }

        # Check for numerical error (test ran but values don't match)
        if "FAIL" in stdout and "max_error" in stdout:
            # Extract max error value
            max_error_match = re.search(r'max_error[=:]?\s*([\d.e+-]+)', stdout)
            max_error = max_error_match.group(1) if max_error_match else 'unknown'
            return False, {
                'type': 'numerical',
                'message': f"Numerical mismatch: max_error = {max_error}",
                'max_error': max_error
            }

        # Runtime error
        if "ERROR:" in stdout or "Exception" in combined_output or "Error" in stderr:
            return False, {
                'type': 'runtime',
                'message': combined_output[-2000:]
            }

        # Unknown failure
        return False, {
            'type': 'unknown',
            'message': combined_output[-2000:]
        }

    except subprocess.TimeoutExpired:
        return False, {
            'type': 'timeout',
            'message': 'Test timed out after 60 seconds'
        }
    except Exception as e:
        return False, {
            'type': 'exception',
            'message': str(e)
        }


def process_function(func_name: str, func_spec: dict) -> dict:
    """Process a single TSVC function with retry logic."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {func_name}")
    print(f"  Arrays: {list(func_spec['arrays'].keys())}")
    print(f"  Offset: {func_spec['has_offset']}, Conditional: {func_spec['has_conditional']}, Reduction: {func_spec['has_reduction']}")
    print(f"{'=' * 70}")

    baselines_dir = Path("baselines")
    test10_dir = Path("test10")
    llm_triton_dir = test10_dir / "llm_triton"
    func_code_dir = llm_triton_dir / func_name  # llm_triton/s000/
    func_raw_dir = llm_triton_dir / "raw_responses" / func_name  # llm_triton/raw_responses/s000/
    test_dir = Path("my_triton_implementations") / func_name

    baselines_dir.mkdir(exist_ok=True)
    test10_dir.mkdir(exist_ok=True)
    llm_triton_dir.mkdir(exist_ok=True)
    func_code_dir.mkdir(exist_ok=True)
    (llm_triton_dir / "raw_responses").mkdir(exist_ok=True)
    func_raw_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    # Create __init__.py files to make directories importable
    (test10_dir / "__init__.py").touch()
    (llm_triton_dir / "__init__.py").touch()
    (func_code_dir / "__init__.py").touch()

    baseline_file = baselines_dir / f"{func_name}_baseline.py"
    test_file = test_dir / f"test_{func_name}_correctness.py"

    results = {
        "baseline_generated": False,
        "triton_generated": False,
        "test_generated": False,
        "test_passed": False,
        "attempts": 0,
        "final_error": None
    }

    # Step 1: Generate PyTorch baseline (skip if exists)
    if baseline_file.exists():
        print(f"  Baseline already exists: {baseline_file}")
        results["baseline_generated"] = True
    else:
        print(f"  Generating PyTorch baseline...")
        code = generate_pytorch_baseline(func_name, func_spec)
        if code:
            with open(baseline_file, 'w') as f:
                f.write(code)
            print(f"  Saved to: {baseline_file}")
            results["baseline_generated"] = True
        else:
            print(f"  Failed to generate baseline")
            return results

    # Step 2: Generate Triton with retry loop
    original_prompt = None
    last_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        results["attempts"] = attempt

        # File paths for this attempt
        triton_file = func_code_dir / f"attempt{attempt}.py"
        raw_file = func_raw_dir / f"attempt{attempt}.txt"

        try:
            if attempt == 1:
                # Initial generation
                triton_code, original_prompt, full_response = generate_triton_initial(func_name)
            else:
                # Retry with error feedback
                triton_code, retry_prompt = generate_triton_with_retry(
                    func_name, original_prompt, last_code, error_info, attempt
                )
                full_response = retry_prompt + "\n\n" + "=" * 80 + "\nRESPONSE:\n" + "=" * 80 + "\n" + triton_code

            last_code = triton_code

            # Save raw response
            with open(raw_file, 'w') as f:
                f.write(full_response)

            # Save Triton code
            with open(triton_file, 'w') as f:
                f.write(triton_code)
            print(f"  Saved Triton code to: {triton_file}")
            results["triton_generated"] = True

            # Generate correctness test for this attempt
            test_code = generate_correctness_test(func_name, func_spec, attempt)
            with open(test_file, 'w') as f:
                f.write(test_code)
            test_file.chmod(0o755)
            results["test_generated"] = True

            # Run test
            print(f"  Running correctness test (attempt {attempt}/{MAX_ATTEMPTS})...")
            passed, error_info = run_test(func_name, test_file)

            if passed:
                print(f"  All tests PASSED on attempt {attempt}!")
                results["test_passed"] = True
                return results
            else:
                error_type = error_info.get('type', 'unknown')
                error_msg = error_info.get('message', '')[:200]
                print(f"  Test FAILED: {error_type}")
                print(f"    {error_msg[:100]}...")
                results["final_error"] = error_info

                if attempt < MAX_ATTEMPTS:
                    print(f"  Will retry with error feedback...")
                else:
                    print(f"  Max attempts ({MAX_ATTEMPTS}) reached. Moving to next function.")

        except Exception as e:
            print(f"  Exception during generation: {e}")
            results["final_error"] = {'type': 'generation_error', 'message': str(e)}
            if attempt >= MAX_ATTEMPTS:
                break

    return results


def main():
    """Main automation pipeline."""
    print("=" * 70)
    print("Integrated Generation and Testing Pipeline")
    print(f"Total functions available: {len(TSVC_FUNCTIONS)}")
    print(f"Max attempts per function: {MAX_ATTEMPTS}")
    print("=" * 70)

    if not client:
        print("ERROR: ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Check if specific functions requested
    if len(sys.argv) > 1:
        func_names = sys.argv[1:]
        functions_to_process = {k: TSVC_FUNCTIONS[k] for k in func_names if k in TSVC_FUNCTIONS}
        not_found = [k for k in func_names if k not in TSVC_FUNCTIONS]
        if not_found:
            print(f"Warning: Functions not found: {not_found}")
        print(f"Processing {len(functions_to_process)} specific functions: {list(functions_to_process.keys())}")
    else:
        functions_to_process = TSVC_FUNCTIONS
        print(f"Processing ALL {len(functions_to_process)} functions")

    if not functions_to_process:
        print("No valid functions to process!")
        return

    # Process each function
    all_results = {}
    for i, (func_name, func_spec) in enumerate(functions_to_process.items(), 1):
        print(f"\n[{i}/{len(functions_to_process)}]", end=" ")
        results = process_function(func_name, func_spec)
        all_results[func_name] = results

    # Print summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Function':<12} {'Baseline':<10} {'Triton':<10} {'Passed':<10} {'Attempts':<10}")
    print(f"{'-' * 70}")

    for func_name, results in all_results.items():
        baseline = "Y" if results["baseline_generated"] else "N"
        triton = "Y" if results["triton_generated"] else "N"
        passed = "Y" if results["test_passed"] else "N"
        attempts = str(results["attempts"])

        print(f"{func_name:<12} {baseline:<10} {triton:<10} {passed:<10} {attempts:<10}")

    print(f"{'=' * 70}")

    # Count successes
    total = len(all_results)
    baseline_ok = sum(1 for r in all_results.values() if r["baseline_generated"])
    triton_ok = sum(1 for r in all_results.values() if r["triton_generated"])
    passed = sum(1 for r in all_results.values() if r["test_passed"])
    first_try = sum(1 for r in all_results.values() if r["test_passed"] and r["attempts"] == 1)
    retried = sum(1 for r in all_results.values() if r["test_passed"] and r["attempts"] > 1)

    print(f"\nBaselines generated: {baseline_ok}/{total}")
    print(f"Triton generated: {triton_ok}/{total}")
    print(f"Tests passed: {passed}/{total}")
    print(f"  - Passed on first try: {first_try}")
    print(f"  - Passed after retry: {retried}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
