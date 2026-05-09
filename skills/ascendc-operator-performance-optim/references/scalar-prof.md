# 阶段 6：Scalar — 详细参考

#### 6.1 Vector 化优化（消除标量循环）

**问题模式**：使用 `for` 循环逐元素操作（如 one-hot 编码的循环赋值）

**常见优化方法**：
- 用 `Duplicate` + 单次 `SetValue` 替代循环赋值
- 用 `Exp`、`Log`、`Mul`、`Muls` 等向量化 API 替代逐元素操作

**示例**（one-hot 编码优化）：

**反例** — for 循环，每次通过SetValue进行标量替换操作：：
```cpp
for (size_t classIdx = 0; classIdx < numClass_; ++classIdx) {
    int64_t weight = 0;
    if (labelIdx == classIdx) {
        weight = 1;
    }
    labelOneHotLocal_.SetValue(classIdx, weight);
}
```

**正例** — 使用Duplicate批量替换：
```cpp
Duplicate(labelOneHotLocal_, float(0.0), numClass_);
// 同步保证 Duplicate 完成后再 SetValue
PipeBarrier<PIPE_V>();
TEventID eventIdVToS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
SetFlag<HardEvent::V_S>(eventIdVToS);
WaitFlag<HardEvent::V_S>(eventIdVToS);
labelOneHotLocal_.SetValue(labelIdx, float(1.0));
```

**优化原理**：
- `Duplicate` 是一条向量化指令，硬件并行填充整个 tensor，比循环逐个 `SetValue` 快数十倍
- 循环方式下每个 iteration 都有标量判断和标量赋值开销，Scalar 指令占比极高
- 优化后仅需 1 条 Vector 指令 + 1 条标量赋值，大幅降低 Scalar 占比



#### 6.2 Vector 化优化（消除标量循环）

使用 `DataCopyParams`（blockCount / blockLen / srcStride / dstStride）将间隔
搬运描述为一条 DMA 指令下发，而非用 for 循环逐行调用 `DataCopy`。

**反例** — for 循环，每次仅搬运 2 KB：

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth  = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
// 16 次独立的 2KB 搬运，带宽利用率极低
for (int i = 0; i < imgHeight; i++) {
    DataCopy(tensorIn[i * copyWidth], tensorGM[i * imgWidth], copyWidth);
}
```

**正例** — 单条 DMA 描述符，一次搬运 32 KB：

```cpp
constexpr int32_t copyWidth = 2 * 1024 / sizeof(float);
constexpr int32_t imgWidth  = 16 * 1024 / sizeof(float);
constexpr int32_t imgHeight = 16;
DataCopyParams copyParams;
copyParams.blockCount = imgHeight;                     // 16 行
copyParams.blockLen   = copyWidth / 8;                 // 单位: 32B DataBlock
copyParams.srcStride  = (imgWidth - copyWidth) / 8;    // src 行间间隔
copyParams.dstStride  = 0;                             // dst 连续写入
DataCopy(tensorGM, tensorIn, copyParams);
```
**优化原理**：
- stride 方式下发一条 DMA 指令，硬件自主完成全部搬运，可充分利用带宽。
- for 循环方式下发 16 条小 DMA 指令，每条之间还有 Scalar 开销。