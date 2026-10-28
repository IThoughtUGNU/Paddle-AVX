# Paddle-AVX

Unofficial AVX-baseline CPU builds of PaddlePaddle 3.3.1 for Linux x86_64.
The build keeps AVX enabled and disables baseline AVX2, FMA and AVX512.
OpenBLAS uses a fixed, single-threaded Sandy Bridge target. MKL, oneDNN, GPU backends and
several optional accelerators are disabled by this profile.

## Install

Install the release wheel into a Python 3.12 virtual environment:

```sh
python -m pip install '<WHEEL_PATH>'
python packaging/avx/smoke.py
```

The initial binary targets an Ubuntu 24.04-compatible environment and references
GLIBC 2.38 and GLIBCXX 3.4.32 symbols. Its `linux_x86_64` tag is not a manylinux
compatibility claim. AVX support is required; older non-AVX CPUs are unsupported.

## Build

Use a Linux x86_64 builder with a C/C++ compiler, Git, Make, Python 3.12 development
headers and zlib development headers. GCC 13 is the tested compiler.

```sh
python3.12 -m venv ../paddle-avx-tools
. ../paddle-avx-tools/bin/activate
python -m pip install --only-binary=:all: -r packaging/avx/requirements-build.txt
PADDLE_VERSION=3.3.1 cmake -S . -B ../paddle-avx-build -G Ninja \
  -C cmake/avx_cpu.cmake -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
MAKE_NB_JOBS=1 cmake --build ../paddle-avx-build --target paddle_copy --parallel 2
```

Upstream configuration acquires and patches its third-party sources. Git and
HTTPS access to those upstream dependencies are required. Choose parallelism
according to available RAM; generated C++ files can use several GB per job.
Keep build directories and downloaded dependencies out of source commits.

The wheel is written under `../paddle-avx-build/python/dist/`. Add a distinct
build identifier with `python -m wheel tags --build 1avxcpu '<WHEEL_PATH>'`.
Preserve upstream licensing and bundled dependency notices with distributed
binaries. Record the exact source revision, build settings and wheel checksum
for each release; do not replace an existing published artifact.

## Check

```sh
python packaging/avx/test_profile.py
python packaging/avx/audit_compile_commands.py ../paddle-avx-build/compile_commands.json
```

Install the wheel in a separate environment and run `packaging/avx/smoke.py`.
Repeat the smoke test on actual AVX-only hardware or under an explicitly
configured AVX-only CPU emulator. Running on an AVX2 machine alone is insufficient.
Compiler-flag checks do not prove every native dependency's instruction baseline.
Validate downstream workloads and performance before deployment. The CPU profile
and smoke checks do not establish full training or GPU compatibility.
