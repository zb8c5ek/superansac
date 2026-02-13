# Building pysuperansac on Windows

## Prerequisites

| Dependency | Tested version | Notes |
|---|---|---|
| **Visual Studio Build Tools** | VS 2022+ (MSVC 19.x) | C++ desktop workload with CMake support |
| **CMake** | 3.12+ | Must be on PATH, or provided via conda |
| **Python** | 3.8+ | With development headers (`python3-dev` / conda) |
| **Eigen3** | 3.x | `conda install eigen` or system install |
| **Boost** | any recent | Only `system` component needed |
| **OpenCV** | 4.x | With contrib modules if needed |
| **pybind11** | 3.x | `conda install pybind11` |

All C++ dependencies can be installed via conda/micromamba into the target environment:

```bash
micromamba install -n <env> eigen boost-cpp opencv pybind11 cmake
```

## Install (editable / development mode)

```bash
cd D:\superansac
pip install -e .
```

This runs the full CMake configure + build cycle via `setup.py`, then
registers the package so `import pysuperansac` works. The built `.pyd`
and `superansac_core.dll` are copied into the repo root for the editable
finder to pick up.

## Install (regular)

```bash
pip install .
```

Artifacts are placed directly into `site-packages`; no files are left in
the source tree.

## Verify

```bash
python -c "import pysuperansac; print(dir(pysuperansac))"
```

## MSVC-specific build fixes applied

Three issues had to be resolved for the MSVC toolchain:

### 1. LTCG vs WINDOWS_EXPORT_ALL_SYMBOLS (`src/CMakeLists.txt`)

The global `/GL` flag (Whole Program Optimization) produces LTCG
"anonymous" `.obj` files. CMake's `WINDOWS_EXPORT_ALL_SYMBOLS` uses
`dumpbin /SYMBOLS` to auto-generate a `.def` file for the shared
library, but `dumpbin` cannot parse anonymous objects -- so the `.def`
comes out empty and no symbols are exported, causing `LNK2001` errors
when the pybind module links against `superansac_core.lib`.

**Fix:** `/GL` and IPO are disabled specifically for the `SupeRANSAC`
shared-library target while the pybind module can still use LTO:

```cmake
if(MSVC)
  set_property(TARGET SupeRANSAC PROPERTY INTERPROCEDURAL_OPTIMIZATION OFF)
  set_property(TARGET SupeRANSAC PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
  target_compile_options(SupeRANSAC PRIVATE $<$<CONFIG:Release>:/GL->)
endif()
```

### 2. Extension name mismatch (`setup.py`)

The `ext_modules` entry originally used `'SupeRANSAC'` as the extension
name, but CMake builds the Python module as `pysuperansac`. The
setuptools editable finder maps the extension name to a source-tree
path, so the names must match.

**Fix:** Changed to `CMakeExtension('pysuperansac', sourcedir='.')`.

### 3. Missing DLL and runtime output directory (`setup.py`)

On Windows, CMake treats `.dll` files as **runtime** outputs (governed
by `CMAKE_RUNTIME_OUTPUT_DIRECTORY`), not library outputs. Without
setting this, `superansac_core.dll` was placed in a different location
than the `.pyd`, causing `ImportError: DLL load failed`.

Additionally, MSVC multi-config generators place outputs under a
`Release/` subdirectory. For editable installs, the artifacts must be
copied to the repo root where the finder expects them.

**Fix:** Added `CMAKE_RUNTIME_OUTPUT_DIRECTORY` and a post-build copy
step in `setup.py`:

```python
cmake_args = [
    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + extdir,   # <-- added
    '-DPYTHON_EXECUTABLE=' + sys.executable,
]

# Post-build: copy .pyd/.dll to source root for editable installs
for pattern in ['*.pyd', '*.so', '*.dll']:
    for f in glob(os.path.join(extdir, cfg, pattern)):
        dst = os.path.join(srcdir, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst):
            shutil.copy2(f, dst)
```

## Cleanup

After an editable install, the repo root will contain generated
artifacts that should not be committed:

```
pysuperansac.cp311-win_amd64.pyd
superansac_core.dll
```

These are already covered by a typical `.gitignore` for `*.pyd` / `*.dll`.
