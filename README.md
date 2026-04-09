# Augmenting LLM Code Translation with Compiler Analysis for C to Triton Kernel Generation

Xiao Qin, Chunwei Xia, Zheng Wang (University of Leeds)

## Repository Structure

```
compiler-guided-triton-gen/
│
├── analysis/                        # Stage 1: Compiler analysis
│   ├── kernel_analysis.py                 Unified analysis module (runs all passes,
│   │                                      produces structured JSON, pattern-agnostic)
│   ├── compute_parallel_dims.py           Parallelization dimension analysis
│   ├── compute_war_dependences.py         Write-after-read dependence analysis
│   ├── compute_reduction_type.py          Reduction pattern detection
│   ├── compute_scalar_expansion.py        Scalar expansion for privatization
│   ├── compute_gpu_parallelization_strategy.py  GPU strategy recommendation
│   ├── llvm_analyzer.py                   LLVM DependenceAnalysis integration
│   ├── llvm_fallback_adapters.py          Fallback adapters for LLVM analysis
│   ├── extract_tsvc_kernels.py            Extract TSVC kernels for analysis
│   ├── extract_polybench_kernels.py       Extract PolyBench kernels for analysis
│   ├── kernels/                           Extracted TSVC kernel C files
│   ├── kernels_polybench/                 Extracted PolyBench kernel C files
│   ├── kernels_realworld/                 Extracted Rodinia/ECP kernel C files
│   ├── results/                           Analysis output (JSON)
│   └── legacy/                            13 standalone analysis scripts (not used
│       ├── compute_convolution_pattern.py   by the active pipeline; preserved for
│       ├── compute_dependences.py           reference and potential future use)
│       ├── compute_loop_interchange.py
│       └── ...
│
├── pipeline/                        # Stage 2+3: LLM generation & profiling optimization
│   ├── generate_and_test_polybench.py     PolyBench/C pipeline (unified analysis
│   │                                      + profiling feedback loop)
│   ├── generate_and_test.py               Main TSVC pipeline
│   ├── generate_and_test_rodinia.py       Rodinia pipeline
│   ├── generate_and_test_realworld.py     ECP proxy apps pipeline
│   ├── auto_test_all_tsvc.py              Batch runner for all 151 TSVC kernels
│   ├── benchmark_large_sizes.py           Performance benchmarking (large data)
│   ├── benchmark_large_sizes_ablation.py  Ablation: with vs without analysis
│   ├── benchmark_tsvc_sizes.py            TSVC benchmarking across sizes
│   ├── measure_total_speedup.py           Aggregate speedup measurement
│   ├── ncu_profile.py                     Nsight Compute profiling
│   ├── ncu_profile_kernels.py             Kernel-level NCU profiling
│   ├── nondeterminism_test.py             Nondeterminism testing
│   ├── run_nondeterminism_study.py        Full nondeterminism study
│   ├── test_near_misses.py                Near-miss kernel testing
│   ├── c_reference/                       C reference code + compiled .so libraries
│   ├── utilities/
│   │   ├── tsvc_functions_db.py           TSVC function database
│   │   ├── polybench_functions_db.py      PolyBench function database
│   │   ├── rodinia_functions_db.py        Rodinia function database
│   │   ├── generate_llm_triton.py         LLM Triton code generation
│   │   ├── generate_numpy_reference.py    NumPy reference generation
│   │   ├── c_code_parser.py               C code parser
│   │   ├── extract_baselines.py           Baseline extraction
│   │   └── visualize_results.py           Results visualization
│   └── legacy/
│       └── legacy_prompt_builder.py       870-line pattern-specific prompt builder
│                                          (replaced by kernel_analysis.py)
│
├── results/                         # Experiment results
│   ├── tsvc/
│   │   ├── test1/ ... test29/             29 TSVC experiment iterations
│   │   ├── llm_triton/                    Latest TSVC Triton implementations
│   │   ├── baselines/                     TSVC baseline Triton implementations
│   │   └── benchmarks/                    Individual kernel benchmark scripts
│   ├── polybench/
│   │   ├── my_polybench_tests/            PolyBench correctness test outputs
│   │   ├── polybench_results/             PolyBench benchmark results
│   │   └── polybench_results_scale8x/     PolyBench results at 8x data scale
│   ├── rodinia/
│   │   ├── kernels_rodinia/               Rodinia kernel definitions
│   │   ├── my_rodinia_tests/              Rodinia correctness test outputs
│   │   └── rodinia_results/               Rodinia benchmark results
│   └── realworld/
│       ├── my_realworld_tests/            ECP proxy app test outputs
│       └── realworld_results/             ECP proxy app benchmark results
│
├── benchmarks_src/                  # Raw benchmark source code
│   ├── TSVC_2/                            TSVC benchmark suite
│   ├── polybench-c-4.2.1/                PolyBench/C 4.2.1
│   └── gpu-rodinia/                       Rodinia benchmark suite
│
├── paper/                           # LaTeX paper source
│   ├── main.tex
│   ├── approach.tex
│   ├── setup.tex
│   ├── results.tex
│   └── workflow.tex
│
├── presentation/                    # Presentation slides
│   ├── create_slides.py                   PolyBench results slide generator
│   ├── polybench_pipeline_slides.pptx     PolyBench results slides
│   ├── generate_slides.py                 Literature review slide generator
│   ├── lit_review_slides.pptx             Literature review slides
│   ├── generate_comparison_slides.py      Unified vs legacy comparison slides
│   ├── comparison_slides.pptx             Comparison results
│   ├── generate_profiling_slides.py       Profiling feedback results slides
│   └── profiling_feedback_slides.pptx     Profiling feedback results
│
├── pet                              # PET (Polyhedral Extraction Tool) binary
└── requirements.txt
```

## How It Works

The system operates in three stages:

**Stage 1 -- Compiler Analysis** (`analysis/kernel_analysis.py`):
The unified analysis module runs all analysis passes (parallelization, WAR dependences, reduction detection, scalar expansion, GPU strategy) on a C kernel and produces a single structured JSON representation. This is pattern-agnostic: it reports what constraints exist, not how to handle them. The LLM receives analysis *facts* and decides the implementation strategy.

**Stage 2 -- LLM-Guided Generation** (`pipeline/generate_and_test_polybench.py`):
The analysis JSON is rendered into a structured prompt and sent to an LLM (Claude Sonnet 4), which generates a Triton GPU kernel. The kernel is validated against a C reference implementation. On failure, the error is classified (compilation, numerical, missing barriers, low performance) and a targeted retry prompt is issued (up to 10 attempts).

**Stage 3 -- Profiling-Guided Optimization** (optional, `--profile-feedback`):
After a kernel passes correctness, NVIDIA Nsight Compute (NCU) profiles it and classifies the bottleneck (compute-bound, memory-bound, or latency-bound). The metrics and bottleneck diagnosis are fed back to the LLM, which generates an optimized version. The optimization is re-validated for correctness and only kept if it improves speedup. This loop runs for up to 3 iterations with NCU profile caching to avoid redundant profiling.

## Results

| Configuration | Pass Rate | Median Speedup | Mean Speedup | Kernels >1x |
|---|---|---|---|---|
| No analysis (baseline) | 28/30 (93%) | 1.06x | 1.52x | 14/28 |
| Unified analysis | 30/30 (100%) | 1.40x | 1.90x | 16/30 |
| Unified + profiling feedback | 30/30 (100%) | 1.90x | 2.36x | 19/29 |

PolyBench/C at 1x scale. Profiling feedback improved 15 of 30 kernels, with gains up to 47x on individual kernels. TSVC achieves 1.02x median vs OpenMP GPU offloading on the same GPU. Generalizes to 8 Rodinia + ECP application kernels.

## Usage

```bash
cd pipeline

# Run all 30 PolyBench kernels with analysis
python generate_and_test_polybench.py

# Run specific kernels
python generate_and_test_polybench.py gemm lu jacobi_1d

# With profiling feedback (3 iterations)
python generate_and_test_polybench.py --profile-feedback gemm

# Custom profiling iterations
python generate_and_test_polybench.py --profile-feedback --profile-iterations 5 gemm

# At 8x data scale
python generate_and_test_polybench.py --size-scale 8

# Without analysis (ablation baseline)
python generate_and_test_polybench.py --no-analysis

# With OpenMP multi-threaded C reference
python generate_and_test_polybench.py --omp
```

## Dependencies

- Python 3.8+
- PET (Polyhedral Extraction Tool)
- LLVM 17.0.0 (clang, opt)
- Triton
- PyTorch
- NVIDIA GPU with CUDA support
- NVIDIA Nsight Compute (for profiling feedback)
