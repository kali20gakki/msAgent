# 官方算子工作流与架构类型参数

本文档为 msSanitizer skill 的参考资料，收录**官方算子库/样例库各代码仓的检测工作流**与**架构类型参数获取方式**。

官方算子的第一步"运行环境检查"、第五步"结果分析与修复"、第六步"生成最终报告"和第七步"清理编译选项"与自定义算子工作流完全一致（参见 SKILL.md 第 3 章），本文档只描述各代码仓不同的第二到四步（编译选项适配、编译部署、运行检测），相同的步骤不再赘述。

## 1. 官方算子工作流统一处理原则

**算子运行命令获取**（适用于所有官方算子仓的"运行检测"步骤）：

当算子运行需要参数、但用户未明确提供时，按以下优先级获取运行命令：

1. **优先查 README**：进入算子目录，查看 `README.md`（或 `README_en.md`）中的运行示例，使用示例中的默认参数。
2. **其次分析代码**：若 README 中无运行示例，分析算子工程源码和 `CMakeLists.txt`，根据 Kernel 调用方式（`<<<>>>` 直调或 API 调用）和 `main` 函数参数定义，拼凑运行命令。
3. **提示用户**：无论使用哪种方式，检测结束后均需提示用户当前使用的运行命令，并建议用户后续明确指定参数以获得更准确的检测结果。

> 各仓具体的算子执行命令格式参见对应子章节。已覆盖 ops-transformer / ops-nn / ops-math / ops-cv / asc-devkit / catlass / cann-samples / triton-ascend-kernels / shmem。

## 2. ops-transformer仓

### 2.1 编译选项适配和重编译部署

1. ops-transformer 仓的 build.sh 已内置检测桩注入机制，无需修改 CMakeLists.txt。编译时通过 `--op_debug_config sanitizer` 即可完成 msSanitizer 检测插桩。

2. 进入 ops-transformer 根目录，执行以下命令编译算子：

    ```shell
    bash build.sh --pkg --soc=<soc_version> --ops=<算子名> --op_debug_config sanitizer -j<number_of_threads>
    ```

    > `--soc` 按平台（Atlas A2/A3系列产品及Ascend 950PR&950DT系列产品）取值，见[架构类型参数获取方式](#9-架构类型参数获取方式)。
    > 算子名参考CMakeLists.txt中指定的编译结果名称。
    > `-j` 参数指定编译线程数可以加速编译过程。为保证加编译，需要通过 `nproc` 命令获取 CPU 核心数，然后设置为最大值的一半。
    > `--op_debug_config sanitizer` 参数说明：`sanitizer` 注入 msSanitizer 检测桩（`--cce-enable-sanitizer`），并生成告警定位所需的「文件:行号」调用栈信息。

    若提示如下信息，说明编译成功:

    ```shell
    Self-extractable archive "cann-ops-transformer-custom_linux.${arch}.run" successfully created.
    ```

3. 编译成功后，run包存放于项目根目录的build_out目录下，执行以下命令安装：

    ```shell
    ./build_out/cann-ops-transformer-*linux*.run
    ```

    该命令会将算子安装在`${ASCEND_HOME_PATH}/opp/vendors`路径中，即CANN软件安装目录。

    安装成功后会打印成功信息，并提示需要设置环境变量：

    ```shell
    [ops_custom] [2026-06-26 14:36:14] [INFO] using requirements: when custom module install finished or before you run the custom module, execute the command [ export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH} ] to set the environment path
    SUCCESS
    ```

4. 执行上述提到的环境变量，确保运行时能够找到：

    ```shell
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
    ```

    > 若 CANN 版本 **≤ 9.1.0**（9.1.0 及之前，`--op_debug_config sanitizer` 注入检测桩失效），需直接修改 CANN 包内编译脚本注入 `--cce-enable-sanitizer -g`，版本判断与操作见文末**第 11 节备注**。

### 2.2 运行检测

ops 仓算子有自带 examples 案例，需要在项目根目录下运行对应命令。其通用命令格式为：

```shell
bash build.sh --run_example <算子名> <运行模式> <包模式>
```

如用户未提供运行命令参数，按[统一处理原则](#1-官方算子工作流统一处理原则)获取默认命令。

> **GE 图模式算子**（graph / 产物 `test_geir_*`）不能按上面的 eager 命令检测，其检测方式见文末**第 10 节备注**。

## 3. ops-nn / ops-math / ops-cv仓

三个仓库的构建系统完全一致，均内置 `--mssanitizer` 快捷编译选项，可自动注入 `-g` 和 `--cce-enable-sanitizer`（即 `-sanitizer`）到 kernel 侧编译。同时也支持通过 `--bisheng_flags=` 灵活指定编译器标志。

### 3.1 编译选项适配和重编译部署

**方式一（推荐）：使用 `--mssanitizer` 快捷选项**

进入对应仓库根目录，执行以下命令编译算子：

```shell
bash build.sh --pkg --soc=<soc_version> --ops=<算子名> --mssanitizer -j<number_of_threads>
```

> `--mssanitizer` 等价于自动添加 `-g --cce-enable-sanitizer` 到 kernel 编译选项。`--soc` 按平台（Atlas A2/A3系列产品及Ascend 950PR&950DT系列产品）取值，见[架构类型参数获取方式](#9-架构类型参数获取方式)。

**方式二：使用 `--bisheng_flags=sanitizer` 灵活指定**（用法与 [ops-transformer仓](#2-ops-transformer仓) 相同；与 `--mssanitizer` 互斥，不能同时使用）。

编译成功提示与安装部署步骤与 [ops-transformer仓](#2-ops-transformer仓) 完全一致，参见该节步骤 3~4。

> 若 CANN 版本 **≤ 9.1.0**（9.1.0 及之前，`--mssanitizer` / `--bisheng_flags=sanitizer` 注入检测桩失效），需直接修改 CANN 包内编译脚本注入 `--cce-enable-sanitizer -g`，版本判断与操作见文末**第 11 节备注**。

### 3.2 运行检测

与 ops-transformer 一致，使用 `build.sh --run_example` 命令：

```shell
bash build.sh --run_example <算子名> eager cust
```

如用户未提供运行命令参数，按[统一处理原则](#1-官方算子工作流统一处理原则)获取默认命令。

> **GE 图模式算子**（graph / 产物 `test_geir_*`）不能按上面的 eager 命令检测，其检测方式见文末**第 10 节备注**。

## 4. asc-devkit仓

以asc-devkit仓的简单add算子`asc-devkit/examples/01_simd_cpp_api/00_introduction/01_add/add`为例，asc-devkit仓的算子样例目录结构一般为：

```text
├── add                     // 算子目录
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── add.asc             // Ascend C样例实现 & 调用样例
│   └── README.md           // 样例说明文档
```

### 4.1 编译选项适配和重编译部署

1. 首先进入最底层算子目录 `add`，为 `CMakeLists.txt` 增加 mssanitizer 编译/链接选项（通用规则见 SKILL.md 3.2 节，完整写法可直接参考本 skill `references/sample_memcheck/CMakeLists.txt`）。以 `add_executable(demo ...)` 为例，在对应 target 上追加：

   ```cmake
   # 编译选项（与现有 --npu-arch 等选项并列追加）
   target_compile_options(demo PRIVATE
       $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
       -g
       --cce-enable-sanitizer
   )
   # 链接阶段也需添加
   target_link_options(demo PRIVATE
       --cce-enable-sanitizer
   )
   ```

   > target 名以该目录 CMakeLists.txt 的 `add_executable` 为准（不一定叫 `demo`）；`-g` 与 `--cce-enable-sanitizer`（或 `-sanitizer`）必须同时添加，否则无调用栈。

2. 加完编译选项后，在 `add` 目录下执行以下命令重新编译：

    ```shell
    mkdir -p build && cd build                      # 创建并进入build目录
    cmake -DCMAKE_ASC_ARCHITECTURES=<npu-arch> ..   # cmake
    make -j <number_of_threads>                     # 编译工程
    ```

    > **注意事项**
    > 
    > 1. `<npu-arch>` 按平台（Atlas A2/A3系列产品及Ascend 950PR&950DT系列产品）取值，见[架构类型参数获取方式](#9-架构类型参数获取方式)
    > 2. `-j` 参数可选，指定编译线程数以加速编译过程，推荐通过 `nproc` 命令获取 CPU 核心数，然后设置为最大值的一半。

    编译完成后，会在build目录下生成可执行文件demo（不同的算子可执行文件名称可能不一样，具体参考CMakeLists.txt中定义的名称，下同）。

### 4.2 运行检测

asc-devkit 仓的算子样例可以直接 `./` 运行。算子执行命令为：

```shell
./demo <opt_para>
```

（不同算子的可执行文件名可能不同，以 CMakeLists.txt 中 `add_executable` 指定的名称为准）

如用户未提供运行参数，按[统一处理原则](#1-官方算子工作流统一处理原则)获取默认命令。

## 5. catlass仓

catlass仓的算子样例均存放在 `catlass/examples` 目录下，且样例文件夹命名规则为 `编号_算子名`，如：

```text
├── 00_basic_matmul         // 算子目录，名称为两位数编号 + 算子名
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 样例说明文档
│   └── basic_matmul.cpp    // 主文件
```

以`00_basic_matmul`算子为例，展示catlass仓算子编译和检测运行流程。

### 5.1 编译选项适配和重编译部署

catlass 仓已预置 mssanitizer 编译选项支持，无需手动修改 CMakeLists.txt。`scripts/build.sh` 提供 `--enable_mssanitizer` 快捷选项（内部即传递 `-DENABLE_MSSANITIZER=True`），自动注入 `-g` 和 `--cce-enable-sanitizer` 编译/链接选项。

**方式一（推荐）：使用 `scripts/build.sh` 快捷选项**

在仓库根目录 `catlass/` 下执行：

```shell
bash scripts/build.sh <target> -DCATLASS_ARCH=<npu-arch> --enable_mssanitizer
# 示例：bash scripts/build.sh 00_basic_matmul -DCATLASS_ARCH=3510 --enable_mssanitizer
# 可选 --clean：先清理 build 与 output 目录再编译
```

**方式二：手动 cmake 配置**

```shell
mkdir -p build && cd build
cmake .. -DENABLE_MSSANITIZER=ON -DCATLASS_ARCH=<npu-arch>
make -j <number_of_threads> <target>
```

> **参数说明**：
> 
> - `<npu-arch>`：catlass 使用纯数字格式，按平台（Atlas A2/A3系列产品及Ascend 950PR&950DT系列产品）取值，见[架构类型参数获取方式](#9-架构类型参数获取方式)
> - `<target>`：算子目录名，此处为 `00_basic_matmul`
> - `-j` 参数可选，指定编译线程数以加速编译过程，推荐通过 `nproc` 命令获取 CPU 核心数，然后设置为最大值的一半

预置编译选项的定义位于 `catlass/examples/CMakeLists.txt`：

```cmake
if(DEFINED ENABLE_MSSANITIZER AND ENABLE_MSSANITIZER)
    add_compile_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:-g --cce-enable-sanitizer>")
    add_link_options("SHELL:$<$<COMPILE_LANGUAGE:ASC>:--cce-enable-sanitizer --npu-arch=dav-${CATLASS_ARCH}>")
endif()
```

编译产物位置：方式一位于 `catlass/output/bin/<target>`，方式二位于 `catlass/build/examples/<target>/`，可执行文件名均与 target 一致，此处为 `00_basic_matmul`。

### 5.2 运行检测

在仓库根目录 `catlass/` 下执行：

```shell
# 手动单工具检测
mssanitizer --tool=memcheck -- output/bin/00_basic_matmul 256 512 1024 0 > result.log 2>&1

# 或一键四工具检测
python scripts/run_mssanitizer.py --output-dir examples/00_basic_matmul -- output/bin/00_basic_matmul 256 512 1024 0
```

如用户未提供运行参数，按[统一处理原则](#1-官方算子工作流统一处理原则)获取默认命令。

## 6. cann-samples仓

cann-samples 仓的样例位于 `Samples/<分类>/<story>/` 下（如 `Samples/2_Performance/simt_scatter_story/`），采用递进式 story 结构：`src/` 下每个 `.asc` 文件编译为一个**独立可执行目标**（命名 `<story前缀>_<step>`），`scripts/gen_data.py` 生成输入与 golden，可执行文件运行时自动生成数据、执行 kernel 并校验。

### 6.1 编译选项适配和重编译部署

1. 样例的 CMakeLists.txt 无内置 mssanitizer 选项，需在样例目录的 `CMakeLists.txt` 中找到 `target_compile_options` 的 ASC 编译选项块，追加 `-g --cce-enable-sanitizer`：

    ```cmake
    # 修改前（以 simt_scatter_story 为例，foreach 对样例内所有 .asc 生效）
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3>"
    )

    # 修改后
    target_compile_options(${TARGET_NAME} PRIVATE
        "$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>"
        "$<$<COMPILE_LANGUAGE:ASC>:-O3 -g --cce-enable-sanitizer>"
    )
    ```

2. 在仓库根目录编译（`NPU_ARCH` 为必填项，按平台（Atlas A2/A3系列产品及Ascend 950PR&950DT系列产品）取值，见[架构类型参数获取方式](#9-架构类型参数获取方式)）：

    ```shell
    source ${ASCEND_HOME_PATH}/set_env.sh
    cmake -S . -B build -DNPU_ARCH=dav-3510    # Ascend950；Atlas A2/A3 换用 dav-2201
    cmake --build build --parallel          # 全量编译
    # 或只编译单个 story（target 为 story 目录名）
    cmake --build build --target simt_scatter_story
    ```

    编译产物位于 `./build/Samples/<分类>/<story>/` 下，可执行文件与 target 同名，如 `simt_scatter_0_direct_unique`。

### 6.2 运行检测

在仓库根目录执行，逐个对可执行文件运行检测（样例通常无参数，运行时自动调用 `gen_data.py` 生成数据并校验）：

```shell
mssanitizer --tool=memcheck -- ./build/Samples/2_Performance/simt_scatter_story/simt_scatter_0_direct_unique > result.log 2>&1
```

一个 story 有多个 step 可执行文件时，逐个检测。如样例 README 中给出了带参数的运行示例，按[统一处理原则](#1-官方算子工作流统一处理原则)获取。

## 7. triton-ascend-kernels仓（Triton 算子）

triton-ascend（Triton 编译框架）与 triton-ascend-kernels（基于其构建的高性能算子库）两仓算子均为 **Triton JIT 形态**（`@triton.jit` kernel + Python 封装）：无独立编译步骤、无需修改构建文件，sanitizer 插桩由**环境变量**控制，Triton 编译器在 JIT 编译时自动注入。

| 代码仓 | 可检测目标 | 运行入口 |
|--------|-----------|----------|
| triton-ascend-kernels | pip 算子库 `src/triton_ascend_kernels/<分类>/`（gemm/activation/norm/attention/moe等），测试 `tests/<分类>/test_<算子>.py` | pytest |

### 7.1 环境准备与编译选项适配

```shell
source ${ASCEND_HOME_PATH}/set_env.sh

# 首次使用时安装
pip install -e '.[dev]'                   # triton-ascend-kernels（dev 含 pytest/transformers）

# sanitizer 插桩开关（代替 CMake 编译选项适配步骤）
export TRITON_ENABLE_SANITIZER=1        # 编译器自动注入 --enable-sanitizer=true
export TRITON_DISABLE_LINE_INFO=0       # 默认为 true（禁用行号），置 0 以获取调用栈行号
rm -rf ~/.triton/cache                  # 清 JIT 缓存，否则 kernel 不重新插桩编译
```

> **注意**：
> 
> - Triton 有编译缓存。切换 `TRITON_ENABLE_SANITIZER` 前后或修改算子代码后，均需重新 `rm -rf ~/.triton/cache`，否则 kernel 不会重新插桩编译。
> - triton-ascend-kernels 的 `build.sh`（build/test 命令）是打包和 CI 全量测试用的，检测场景无需执行，`pip install -e .` 即完成部署。

### 7.2 运行检测

用 mssanitizer 拉起 Python 进程（脚本直跑或 pytest 均可）：

```shell
# triton-ascend：教程样例直跑
mssanitizer --tool=memcheck -- python3 third_party/ascend/tutorials/01-vector-add.py > result.log 2>&1

# triton-ascend：pytest 测试用例
mssanitizer --tool=memcheck -- pytest python/test/unit/language/test_core.py -k test_add > result.log 2>&1

# triton-ascend-kernels：检测单个算子的测试文件
mssanitizer --tool=memcheck -- pytest tests/activation/test_swiglu.py > result.log 2>&1

# 用 -k 过滤到具体用例（缩小检测范围，报告更聚焦）
mssanitizer --tool=memcheck -- pytest tests/gemm/test_matmul.py -k "test_matmul_bf16" > result.log 2>&1

# 指定 kernel 名检测（只检测目标 kernel，过滤框架自带算子噪声）
mssanitizer --tool=memcheck --kernel-name=swiglu -- pytest tests/activation/test_swiglu.py > result.log 2>&1
```

如用户未提供运行目标，按[统一处理原则](#1-官方算子工作流统一处理原则)获取：triton-ascend-kernels 依据目标算子名在 `tests/<分类>/` 下找同名测试文件（如 swiglu → `tests/activation/test_swiglu.py`），取其默认参数。

> **注意**：
> 
> - 手动检测时**不要加 `-n` 并行**（pytest-xdist 多进程会干扰 mssanitizer 的进程拦截），串行跑单文件即可。
> - triton-ascend-kernels 的测试用例普遍走 `torch.allclose` 精度比对，kernel 算错不等于会报错——**精度断言失败与 sanitizer 告警是两回事**，结果分析时都需关注。
> - 检测完成后 `unset TRITON_ENABLE_SANITIZER TRITON_DISABLE_LINE_INFO` 恢复环境（对应步骤七"清理编译选项"）。

## 8. shmem仓

shmem 仓为昇腾共享内存通信库，通信算子样例位于 `examples/<demo>/`（如 allgather、user_buffer 等，含 `main.cpp`、`*_kernel.cpp`、`run.sh`）。仓内 `scripts/build.sh` 已内置 `-mssanitizer` 快捷选项，无需手动修改 CMakeLists.txt。

> **前置要求**：需 CANN 9.1.0 及以上；**SDMA/RDMA 相关接口和用例不支持** mssanitizer 检测，选样例时避开。

### 8.1 编译选项适配和重编译部署

在仓库根目录执行：

```shell
# 检测库本体
bash scripts/build.sh -soc_type Ascend950 -mssanitizer     # Ascend 950PR&950DT系列产品
bash scripts/build.sh -mssanitizer                          # Atlas A2/A3系列产品（不带 -soc_type，走默认Atlas A2/A3系列产品）

# 检测 examples 通信算子样例（加 -examples）
bash scripts/build.sh -soc_type Ascend950 -examples -mssanitizer
```

> Ascend 950PR&950DT系列产品上是否联编 `--cce-enable-sanitizer` 由 bisheng 版本决定（构建脚本自动选择）：旧版本 CANN 仅添加 `-g`，此时 AscendC API 相关内存检测不可用，如需该能力请升级 CANN 后重新编译。

编译产物：可执行文件位于 `build/bin/<example>`，库位于 `build/lib/`。

### 8.2 运行检测

shmem 样例通过 `run.sh` 拉起（内部会启动多个进程分占多卡）。用 mssanitizer 包裹 run.sh 运行，在仓库根目录或样例目录执行均可（建议在仓库根目录）：

```shell
# 内存检测（默认 memcheck）
mssanitizer -- bash examples/allgather/run.sh -pes 2 > result.log 2>&1

# 卡间竞争检测（shmem 多卡场景常用）
mssanitizer --tool=racecheck --check-cross-npu-races=yes -- bash examples/allgather/run.sh -pes 2 > result.log 2>&1
```

`-pes <N>` 指定进程数（卡数），其他参数见各样例 README / run.sh。

> **结果分析注意**：`aclshmem_malloc` 等 shmem 内存分配接口是从已完成物理地址映射的大块连续虚拟内存中划分，若越界访问的地址恰好落在已映射区域内，工具**不会报错**（该地址合法可用）。分析"未检出问题"或判断误报时需考虑此特性。

## 9. 架构类型参数获取方式

> 本节即 `--soc` / `--npu-arch` / `CATLASS_ARCH` / `NPU_ARCH` / `-soc_type` 等各仓架构参数的统一取值说明。上文各仓步骤引用的"架构类型参数获取方式"均指本节。

首先执行以下命令获取芯片型号：

```shell
python3 -c "import acl; print(acl.get_soc_name())"
```

回显结果列出了芯片类型，如Ascend910B4、Ascend950PR等。根据芯片型号，各代码仓的架构参数取值如下：

| NPU Name | 产品系列 | `--soc`<br>(ops-transformer/nn/math/cv) | `--npu-arch`<br>(asc-devkit) | `CATLASS_ARCH`<br>(catlass) | `NPU_ARCH`<br>(cann-samples) | `-soc_type`<br>(shmem) |
|----------|---------|---------------------|------------------|----------------------|----------------------|------------------|
| Ascend910BX | Atlas A2系列产品 | `ascend910b` | `dav-2201` | `2201` | `dav-2201` | 不指定（默认Atlas A2/A3系列产品 |
| Ascend910_93XX | Atlas A3系列产品 | `ascend910_93` | `dav-2201` | `2201` | `dav-2201` | 不指定（默认Atlas A2/A3系列产品）|
| Ascend950PR/DT | Ascend 950PR&DT系列产品 `ascend950` | `dav-3510` | `3510` | `dav-3510` | `Ascend950` |

> **使用要点**：
> 
> - 各仓参数名与取值格式均不同（如 `dav-2201` vs `2201` vs `ascend910b`），请严格按表头对应仓取用。
> - **A2 与 A3 仅 ops 系列的 `--soc` 区分**（`ascend910b` vs `ascend910_93`）；在 asc-devkit / catlass / cann-samples 上两平台取值相同。
> - **shmem 只区分 950 与非 950**：不带 `-soc_type` 时默认 Atlas A2/A3系列产品（官方注释说明其覆盖A2/A3系列）。
> - **cann-samples 的 `NPU_ARCH` 为必填项**，缺省会直接报错，合法值仅 `dav-3510` / `dav-2201` 两个。

---

## 10. 备注：ops 系列仓 GE 图模式算子的检测

ops 系列仓（ops-transformer / ops-nn / ops-math / ops-cv）中 **GE 图模式算子**（`--run_example` 用 `graph`、产物为 `build/test_geir_<算子名>`）的检测方式与普通 eager 算子不同，**不能**用各仓“运行检测”一节的 eager 命令：

- **原理**：图模式下算子 kernel 下沉执行，检测运行阶段**不做实时检测**，只通过流上 callback 任务把算子输入输出 dump 到磁盘；真正的检测在**算子进程退出后**由工具读取 dump 数据并回放 kernel 完成。一次检测命令会自动执行两遍（第一遍 dump，第二遍用 dump 检测），结束后 dump 自动删除。

步骤：

1. **编译安装**：与普通算子一致（ops-nn/math/cv 用 `--mssanitizer`，ops-transformer 用 `--op_debug_config sanitizer`），编译后安装 run 包并 `export LD_LIBRARY_PATH`。
2. **（可选）改 deviceId**：graph 示例源文件（该仓 examples 下对应算子的 `test_geir_<算子名>.cpp`）中 `deviceId = 0;` 若指向被占用的卡，改为空闲卡号；部分仓在特定 arch 下 `build.sh --run_example` 只查示例的 `arch35` 子目录（若存在）。
3. **生成并验证 graph 示例**：`bash build.sh --run_example <算子名> graph --soc=<soc>`。出现类似 `Run test_geir_<算子名> success.` 即单独运行成功，并生成 `build/test_geir_<算子名>`；失败则按“单独运行失败”处理，不进入检测。
4. **四类检测**：**进入构建产物目录**（如 `<ops仓库>/build`）后直接对生成的二进制运行，每个命令自动跑两遍（dump → 用 dump 检测）：

   ```shell
   cd <ops仓库>/build
   mssanitizer -t memcheck  --log-level=error -- ./test_geir_<算子名> > <结果目录>/memcheck.log  2>&1
   mssanitizer -t racecheck --log-level=error -- ./test_geir_<算子名> > <结果目录>/racecheck.log 2>&1
   mssanitizer -t initcheck --log-level=error -- ./test_geir_<算子名> > <结果目录>/initcheck.log  2>&1
   mssanitizer -t synccheck --log-level=error -- ./test_geir_<算子名> > <结果目录>/synccheck.log  2>&1
   ```

   工具日志/产物落于执行目录（`build/` 下的 `mindstudio_sanitizer_log/`），结果文件可用绝对路径重定向。
5. **判读**：运行阶段只有算子自身 INFO 日志、尚无 `Start xxxcheck` 属正常（正在 dump）；算子进程退出后才出现 `[mssanitizer] Start xxxcheck sanitizer on kernel ...` 与 `Sanitizer finished`（时间晚于算子日志属正常）。进程退出后仍无该日志 → dump 未被消费，视为工具未识别到算子（检测失败）。
6. **产物收集**：把 `mindstudio_sanitizer_log/` 下工具日志拷贝到结果目录，连同四份检测日志一起归档。

> 判定：明确为图模式的算子用 `graph`（产物 `test_geir_*`），普通算子用 `eager`（产物 `test_aclnn_*`）。

---

## 11. 备注：老 CANN（CANN ≤ 9.1.0）下 ops-xx 仓检测编译选项的注入

**适用**：CANN 版本 **≤ 9.1.0（9.1.0 及之前）** 时，ops 系列仓（ops-transformer / ops-nn / ops-math / ops-cv）`build.sh` 的 `--mssanitizer` / `--bisheng_flags=sanitizer` **无法**把检测桩选项注入 kernel 编译（旧工具链未透传该参数），需**直接修改 CANN 安装包内编译脚本**完成注入；CANN 9.1.0 之后（> 9.1.0）无需此步。

**0. 判断 CANN 版本**：查看 CANN 安装目录下 `cann` 软链接实际指向的目录名（命名形如 `cann-<版本>`），从目录名读取版本号：

   ```shell
   readlink -f <CANN安装目录>/cann
   # 如：
   #   readlink -f /home/<user>/cann/730/cann  → /home/<user>/cann/730/cann-9.1.0
   ```

   返回目录名中的版本号 **≤ 9.1.0** 时按本备注手动注入；> 9.1.0 无需。

1. **定位文件**：`<CANN包根>/python/site-packages/asc_op_compile_base/asc_op_compiler/ascendc_compile_v220.py`（CANN 包根如 `/usr/local/Ascend/ascend-toolkit/<版本>` 或 `cann/<版本>`；找不到可 `find <CANN包根> -name ascendc_compile_v220.py`）。
2. **确定修改哪个函数**：按目标芯片选择（判断见第 9 节架构类型参数表）：
   - Ascend 950PR&950DT系列产品 → 修改 `_gen_compile_cmd_c310`；
   - Atlas A2/A3系列产品 → 修改 `_gen_compile_cmd_v220`。
3. **追加检测选项**：在该函数内 `compile_cmd` 的 **`-mllvm` 系列选项之后**追加一行：

   ```python
   # 追加 msSanitizer 检测编译选项（CANN ≤ 9.1.0 手动注入）
   compile_cmd += ["--cce-enable-sanitizer", "-g", "-fno-jump-tables"]
   ```

   （`compile_cmd` 为列表，参照该文件中其它 `compile_cmd += [...]` 的写法即可。）
4. **重新编译**：按对应 ops 仓“编译选项适配”章节重新执行 `build.sh --pkg` 编译，后续安装、运行检测流程不变。

> **注意**：此操作修改 CANN 安装包内文件，升级/重装 CANN 会被覆盖；检测完成后建议还原该文件（或重装对应 CANN），以免影响后续正常编译。
