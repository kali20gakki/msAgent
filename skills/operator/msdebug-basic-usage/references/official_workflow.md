# 官方算子仓 msDebug 调试工作流

本文档为 msDebug skill 的参考资料，收录**官方算子库/样例库各代码仓的调试工作流**：拉起 msdebug 前的编译选项适配（`-g -O0`）、编译部署、可执行文件（msdebug target）获取方式，以及架构参数取值。

msDebug 调试编译**只需要** `-g -O0`（调试信息 + 零优化），**不要**混入检测/插桩类编译标志（会改变运行行为与时序）。各仓编译标志的注入机制也不同（有的仓支持 `--op_debug_config`，有的只支持 `--bisheng_flags=`，有的需改 CMakeLists），本文档以各仓源码实际核实为准。

## 覆盖范围

| 仓库 | `-g -O0` 注入方式 |
|------|-------------------|
| ops-transformer | `build.sh --op_debug_config "ccec_O0,ccec_g"` |
| ops-nn / ops-math / ops-cv | `build.sh --bisheng_flags=ccec_g,ccec_O0` |
| asc-devkit | `cmake -DCMAKE_BUILD_TYPE=Debug`（自动映射 ASC `-O0 -g`） |
| catlass | `scripts/build.sh --debug <target>`（当前版本已支持，零改动） |
| cann-samples | 手动改样例 CMakeLists 的 `-O3` 为 `-O0 -g` |

> 不在本档范围：triton-ascend-kernels（Triton JIT，行号信息由编译器注入开关控制，机制不同）、shmem（通信库多卡场景）。

## 1. 统一处理原则

### 1.1 msdebug target（可执行文件）的三个来源

msdebug 拉起时需要一个宿主侧可执行文件或 fatbin 作为 target，按仓库类型分三种：

1. **ops 系列仓**（transformer/nn/math/cv）：`build.sh --run_example <算子名> eager cust` 生成 `test_aclnn_<算子名>`（graph 模式为 `test_geir_<算子名>`），产物在各仓 **`build/`** 目录下（不是 build_out）。
2. **样例型仓库**（asc-devkit / catlass / cann-samples）：每个样例编译出独立可执行文件（build 目录或 output/bin 下），直接作为 target。
3. **Kernel 直调场景**：可执行文件内嵌 fatbin，直接 `msdebug ./app`。

### 1.2 kernel 侧 `-g -O0` 编译

各仓注入方式见覆盖范围表与后续章节。公共注意事项：

- `-O0` 会改变指令调度，某些依赖优化行为的 bug（如乱序踩内存）在 `-O0` 下可能不复现，必要时改用不带 `-O0` 的编译对比行为差异。
- 部分仓 kernel 内函数被强制 inline 时行号可能缺失（Ascend 950PR&950DT系列产品的simd_vf 场景），处理方式见 SKILL.md 第 8 章。
- 编译选项是临时修改的场景，调试完成后恢复原始状态重新编译。

### 1.3 算子运行参数获取

当算子运行需要参数、用户未明确提供时：

1. **优先查 README**：算子/样例目录下的 `README.md` 运行示例。
2. **其次看 run_example / gen_data**：ops 系列仓 examples 自带默认参数；cann-samples 样例运行时自动调 `gen_data.py` 生成数据。
3. **提示用户**：调试结束后告知实际使用的运行命令。

### 1.4 部署与环境变量（ops 系列仓）

ops 系列仓编译产出 vendor run 包（`build_out/` 下），需安装后设置 `LD_LIBRARY_PATH` 才能运行 `test_aclnn_xxx`：

```shell
# 各仓安装（包名模式见各章节）
./build_out/cann-ops-<仓>*linux*.run
# 各仓环境变量（<vendor> 为安装时指定的 vendor 名，默认 custom 系列）
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/<vendor>/op_api/lib:${LD_LIBRARY_PATH}
```

## 2. ops-transformer 仓

### 2.1 编译选项适配和重编译部署

build.sh 内置 `--op_debug_config` 参数（cmake/func.cmake 中 `add_opc_config` 将 `ccec_g`/`ccec_O0` 映射为 kernel 侧 `-g`/`-O0`），无需修改任何文件：

```shell
cd <ops-transformer根目录>
bash build.sh --pkg --ops=<算子名> --soc=<soc_version> --op_debug_config "ccec_O0,ccec_g" -j<线程数>
```

> - 大算子（tilingKey 多）编译失败或运行即精度异常时，追加 `--tiling_key=<tilingKey>`（确定方法：先跑一次观察 `[Launch of Kernel <算子>_<tilingKey>_<x> on Device N]` 中的后缀）。
> - 也可用 `--bisheng_flags=ccec_g,ccec_O0`（与 `--op_debug_config` 等价路径，二选一）。

编译成功回显 `Self-extractable archive "cann-ops-transformer-custom_linux.<arch>.run" successfully created.`，安装：

```shell
./build_out/cann-ops-transformer-*linux*.run
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
```

### 2.2 拉起 msdebug

```shell
# run_example 生成宿主侧可执行文件（会先自动运行一次，正常退出即可）
bash build.sh --run_example <算子名> eager cust
# 可执行文件位于 <根目录>/build/test_aclnn_<算子名>，拉起调试：
msdebug ./build/test_aclnn_<算子名>
```

## 3. ops-nn / ops-math / ops-cv 仓

三仓构建系统同源，**均不支持 `--op_debug_config` 命令行参数**（该名只是内部传给 CANN `asc_opc` 工具的选项）。正确的注入入口是 `--bisheng_flags=`，其值被原样透传为 `asc_opc --op_debug_config=<值>`，`ccec_g`/`ccec_O0` 到 `-g`/`-O0` 的翻译在 CANN 工具链内完成：

```shell
cd <ops-nn|ops-math|ops-cv>根目录
bash build.sh --pkg --soc=<soc_version> --ops=<算子名> --bisheng_flags=ccec_g,ccec_O0 -j<线程数>
```

> - 编译线程数推荐 `nproc` 的一半。
> - 三仓均**不支持 `--tiling_key` 命令行参数**（仅为内部 asc_opc 选项）。大算子调试受限时，考虑改用 UT 编译入口（`tests/ut/op_kernel` 的 AddOpTestCase，可传自定义编译选项）或反馈补参数。

编译成功后安装与 ops-transformer 一致（包名模式 `cann-ops-<nn|math|cv>*linux*.run`，位于 `build_out/`）：

```shell
./build_out/cann-ops-<nn|math|cv>*linux*.run
# 环境变量（vendor 名以安装回显为准）
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/<vendor>_nn/op_api/lib:${LD_LIBRARY_PATH}   # ops-nn
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/<vendor>_math/op_api/lib:${LD_LIBRARY_PATH} # ops-math
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/<vendor>_cv/op_api/lib:${LD_LIBRARY_PATH}   # ops-cv
```

### 3.1 拉起 msdebug

```shell
# 生成宿主侧可执行文件（eager 模式），产物在各仓 build/ 目录下
bash build.sh --run_example <算子名> eager cust
# ops-math 支持 --noexec 跳过自动运行；ops-nn/ops-cv 会自动运行一次，正常退出后手动拉起调试
msdebug ./build/test_aclnn_<算子名>
```

> 细节差异：ops-nn 支持 `--example_name=` 选择单个示例，ops-cv 不支持；ops-nn/ops-math 的 eager 可执行文件为 `test_aclnn_<example>`，graph 模式为 `test_geir_<example>`。

## 4. asc-devkit 仓

每个样例目录独立 cmake 构建（顶层 build.sh 只构建库本身，与 examples 无关）。**该仓对调试编译支持最好**：ASC 语言的 per-config 默认 flags 中 `Debug` 配置即 `-O0 -g`（cmake/asc/asc_modules/CMakeASCInformation.cmake 的 `CMAKE_ASC_FLAGS_DEBUG_INIT "-O0 -g"`），只需加 `-DCMAKE_BUILD_TYPE=Debug`，零文件改动：

```shell
cd <样例目录，如 examples/01_simd_cpp_api/00_introduction/04_reg_compute/add>
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_ASC_ARCHITECTURES=<npu-arch> ..
make -j <线程数>
```

> 等价注入方式（任选其一）：`cmake -DCMAKE_ASC_FLAGS="-g -O0" ..`；或 `export ASCFLAGS="-g -O0"` 后正常 cmake。
> **务必使用默认的 npu 运行模式**（`CMAKE_ASC_RUN_MODE` 不设或设为 `npu`）；cpu/sim 模式不跑在真实 NPU 上，msdebug 上板调试无意义。
> 注意各样例 CMakeLists 中可能写死 `--cce-simd-vf-fusion=false` 等选项，保持不动即可。

### 4.1 拉起 msdebug

可执行文件在样例 build 目录下（名称以 CMakeLists 的 `add_executable` 为准，多为 `demo`）：

```shell
python3 ../scripts/gen_data.py    # 生成输入数据（样例需要时）
msdebug ./demo                    # 拉起调试
# 运行期参数按各样例 README；结果校验：python3 ../scripts/verify_result.py output/output.bin output/golden.bin
```

## 5. catlass 仓

catlass 当前版本（已核对本地源码）已原生支持调试编译参数注入，**无需改任何文件**：`scripts/build.sh --debug` 透传 `-DCMAKE_BUILD_TYPE=Debug`，使 ASC 设备侧与宿主侧一并带上 `-O0 -g`（catlass 的 CMake 层不写死 ASC `-O`，优化级来自 CANN ASC 工具链 per-config 默认值：Debug 即 `-O0 -g`）。另有 `--msdebug`（透传 `-DASCEND_ENABLE_MSDEBUG=True`）可配合调试场景使用。

编译与拉起（以 `examples/00_basic_matmul` 为例）：

```shell
cd <catlass根目录>
source ${ASCEND_HOME_PATH}/set_env.sh
bash scripts/build.sh --debug 00_basic_matmul -DCATLASS_ARCH=<catlass_arch> [--clean]
# 可执行文件：output/bin/00_basic_matmul
msdebug ./output/bin/00_basic_matmul -- 256 512 1024 0    # 参数 m n k deviceId
```

> Ascend 950PR&950DT系列产品样例（如 `43_ascend950_basic_matmul`）需 `-DCATLASS_ARCH=3510`；部分样例仅支持特定 arch（见 examples/CMakeLists.txt 的 2201/3510 分组）。
> 若所用 catlass 版本较旧、`scripts/build.sh` 尚无 `--debug` 开关，可退回手动方式：在 `examples/CMakeLists.txt` 的 ASC 编译选项处追加 `add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-O0 -g>")`，调试完成后删除该行并恢复文件原始状态。

## 6. cann-samples 仓

样例位于 `Samples/<分类>/<story>/`，ASC 优化级别写死在各 story 的 CMakeLists.txt（`target_compile_options` 的 `$<$<COMPILE_LANGUAGE:ASC>:-O3>`）。将其改为 `-O0 -g`：

```cmake
# 修改前（以 vector_add 为例）
target_compile_options(vector_add PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
)
# 修改后
target_compile_options(vector_add PRIVATE
    "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
    "$<$<COMPILE_LANGUAGE:ASC>:-O0 -g>"
)
```

> 部分 story 用集中变量（如 matmul_story 的 `MATMUL_ASCENDC_COMPILE_OPTS`），改一处即可覆盖全部子 target。
> `NPU_ARCH` 为**必填**，合法值仅 `dav-3510`（Ascend 950PR&950DT系列产品）/ `dav-2201`（Atlas A2/A3系列产品）；每个 story 在 CMakeLists 内联 arch 门禁（`if(NOT "${NPU_ARCH}" IN_LIST SUPPORTED_NPU_ARCHS) ... return()`），不支持的 arch 会跳过该样例。

编译与拉起：

```shell
cd <cann-samples根目录>
source ${ASCEND_HOME_PATH}/set_env.sh
cmake -S . -B build -DNPU_ARCH=<npu_arch>
cmake --build build --target <story或target名> --parallel <线程数>
# 构建树产物：build/Samples/<分类>/<story>/<target名>；或 install 后 build_out/<相对路径>/<target名>
msdebug ./build/Samples/0_Introduction/vector_add/vector_add
```

> 样例运行通常无参数（自动调 gen_data.py 生成数据并校验）；带参数的按样例 README。调试完成后恢复 CMakeLists 的 `-O3` 并重编。

## 7. 架构类型参数获取方式

先获取芯片型号：

```shell
python3 -c "import acl; print(acl.get_soc_name())"
```

按芯片型号取各仓参数：

| NPU Name | 产品系列 | `--soc`<br>(ops-transformer/nn/math/cv) | `CMAKE_ASC_ARCHITECTURES`<br>(asc-devkit) | `CATLASS_ARCH`<br>(catlass) | `NPU_ARCH`<br>(cann-samples) |
|----------|---------|---------------------|------------------|----------------------|------------------|
| Ascend910BX | Atlas A2系列产品 | `ascend910b` | `dav-2201` | `2201` | `dav-2201` |
| Ascend910_93XX | Atlas A3系列产品 | `ascend910_93` | `dav-2201` | `2201` | `dav-2201` |
| Ascend950PR/DT | Ascend 950PR&950DT系列产品 | `ascend950` | `dav-3510` | `3510` | `dav-3510` |

> - 各仓参数名与取值格式不同（`ascend910b` vs `dav-2201` vs `2201`），严格按表头对应仓取用。
> - A2 与 A3 仅 ops 系列的 `--soc` 区分；asc-devkit/catlass/cann-samples 两平台取值相同。
