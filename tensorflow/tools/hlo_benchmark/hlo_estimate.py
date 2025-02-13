import subprocess
import glob
import re
import argparse

# Paths to the input and output files
parser = argparse.ArgumentParser(description="""Generate Tensile config file""")

parser.add_argument(
    "--hlo",
    type=str,
    help="Glob path to hlo modules")

parser.add_argument(
    "--output",
    type=str,
    help="Output file path")

parser.add_argument(
    "--warmup", type=int, default=10,
    help="Warmup iterations")

parser.add_argument(
    "--iters", type=int, default=10,
    help="Max tuning iterations")

args = parser.parse_args()


# PATH = "/home/sixifang/tensorflow-upstream/bubble_test_xla_dump/*.gfx90a_gpu_after_optimizations.txt"
# OUTPUT_FILE = "result.txt"
PATH = args.hlo
OUTPUT_FILE = args.output

HLO_BENCH_RE = r"execution time for runner ROCM: (?P<TIME>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)"
FLOPS_RE = (r"(?P<FLOPS>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) GFLOPS. "
           r"(?P<BYTES>[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?) MiB.")


files = glob.glob(PATH)
with open(OUTPUT_FILE, 'w') as f:
    for file in files:
        res = subprocess.run(f"bazel-bin/tensorflow/compiler/xla/tools/run_hlo_module --reference_platform='' --xla_disable_all_hlo_passes=true --iterations={args.warmup+args.iters} --platform=gpu {file}", shell=True, capture_output=True)
        lines = res.stderr.decode('utf-8').split('\n')
        times = []
        for line in lines:
            match = re.search(
                HLO_BENCH_RE, line
            )
            
            if match:
                time = float(match.group('TIME').strip())
                times.append(time)
        times = times[args.warmup:]
        avg_time = 0
        if len(times)>0:
            avg_time = sum(times) / len(times)

        res = subprocess.run(f"bazel-bin/tensorflow/compiler/xla/tools/compute_cost --format=hlo --input={file}", shell=True, capture_output=True)
        lines = res.stdout.decode('utf-8').split('\n')
        match = re.search(FLOPS_RE, lines[0])
        flops = 0
        bytes = 0
        for line in lines:
            match = re.search(FLOPS_RE, line)
            if match:
                flops = float(match.group('FLOPS').strip())
                bytes = float(match.group('BYTES').strip())
                break
        tflops = 0
        if avg_time > 0 and flops > 0:
            tflops = flops / avg_time

        f.write(f"{file}   {tflops} GFLOPS/s   {avg_time} s   {bytes} MiB\n")
        f.flush()
