# A script about how to compute HLO Module FLOPS.
## Build
```
bash install_rocm.sh tensorflow/compiler/xla/tools:run_hlo_module tensorflow/compiler/xla/tools:compute_cost
```

## Usage
```
python tensorflow/tools/hlo_benchmark/hlo_estimate.py --hlo=tensorflow/compiler/xla/tests/*.gfx942_gpu_after_optimizations --output=result.txt
```

## Example Output
slow_xla_sample_v2/module_5629.cluster_4221__XlaCompiledKernel_true__XlaHasReferenceVars_true__XlaNumConstantArgs_0__XlaNumResourceArgs_0_.785.gfx942_gpu_after_optimizations.txt   4332.0634140346565 GFLOPS/s   0.0008137 s