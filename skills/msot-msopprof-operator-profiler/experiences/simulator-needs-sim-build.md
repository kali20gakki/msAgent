# 经验案例：部分工程的仿真拉起需要仿真兼容构建产物

> 这是一条 **经验案例**，不是 `msprof op simulator` 的无条件通用规则。  
> 结论应该表述为：**某些工程 / 模板库 / 构建链在仿真场景下需要单独生成“仿真兼容”的可执行文件**，而不是简单说“所有仿真都必须 sim 编译”。

## 适用症状

用户用已有可执行文件直接拉起仿真时，报类似错误：

```text
terminate called after throwing an instance of 'std::__ios_failure'
  what():  basic_filebuf::xsgetn error reading the file: Bad address
[WARN]  Child process killed by signal 6
```

## 为什么这条经验不是“绝对规则”

仓内官方文档同时给出了两类信号：

1. **有些工程明确需要 simulator 构建**
   - 例如 catlass / 模板库场景，官方文档明确要求构建脚本加 `--simulator`。
2. **也有些官方示例说明同一可执行文件可在设备和仿真器上运行**
   - 这说明“是否必须单独仿真构建”取决于工程的构建链和产物形式，而不是 msOpProf 的统一硬性规则。

因此，遇到此类错误时，更稳妥的根因表述是：

> 当前拉起的可执行文件可能**不是仿真兼容产物**，或者缺少与仿真器匹配的构建选项 / 依赖配置。

## 典型根因

常见于以下场景：

- 工程区分 device / simulator 两类构建产物；
- 模板库或脚本要求显式 `--simulator`；
- 可执行文件默认只面向设备运行时；
- 仿真器所需依赖库、架构参数或链接方式与当前二进制不匹配。

## 建议排查顺序

### 1. 先看工程文档或构建脚本

优先检查是否存在这些信号：

- `--simulator`
- `sim`
- `*_sim`
- 独立的 simulator target / profile
- 单独的架构或运行模式参数

### 2. 再看仿真器环境

确认：

```bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
```

其中 `Ascendxxxyy` 要与实际仿真器类型一致。

### 3. 如构建链支持，生成仿真兼容产物再试

#### 示例 A：模板库 / catlass 风格

```bash
bash scripts/build.sh --simulator 00_basic_matmul
```

#### 示例 B：项目级 CMake 开关（示例，不是 msOpProf 固定接口）

```bash
cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..
make -j
```

> 注意：这类 CMake 变量是**项目构建链示例**，不是 `msprof op simulator` 自身参数；是否支持取决于用户工程。

## 验证标准

满足以下多数信号时，基本可以判断问题已解决：

- 不再出现 `signal 6` / `Bad address`
- 日志能进入正常仿真执行阶段
- 输出目录下能正常生成 `dump/` 与 `simulator/`
- `simulator/` 下能看到每核子目录、`trace.json`、`visualize_data.bin` 等产物

## 不适用的场景

以下场景不应直接套用“需要 sim 编译”结论：

1. 用户走的是 `--config` 或 `--export`，本身就不依赖同一个 app 拉起路径。
2. 官方示例或工程文档已经明确说明同一产物可同时运行在设备和仿真器上。
3. 根因其实是：
   - `LD_LIBRARY_PATH` 指错仿真器
   - `--soc-version` 与产物不匹配
   - dump / 输出目录权限问题
   - 缺少 `aicore_binary.o`

## 案例元数据

- 日期：2026-04-22
- 芯片：Ascend910B4
- CANN 版本：9.0.0
- 算子：MatmulLeakyRelu（Ascend C）
- 经验结论：对该工程而言，改用仿真兼容构建产物后，仿真恢复正常
