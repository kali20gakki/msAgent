# 数据集配置映射

AISBench 注册名速查表。

## 常用数据集

| 数据集 | config_name | 说明 |
|--------|-------------|------|
| gsm8k | `gsm8k_gen_0_shot_cot_str` | 数学推理 |
| aime25 | `aime2025_gen_0_shot_chat_prompt` | 高阶数学 |
| aime24 | `aime2024_gen_0_shot_chat_prompt` | 高阶数学 |
| bfcl-simple | `BFCL_gen_simple` | 工具调用 |
| bfcl-hard | `BFCL_gen_hard` | 工具调用（难） |
| mmlu | `mmlu_gen_0_shot` | 知识问答 |
| mmlu-pro | `mmlu_pro_gen_0_shot` | 知识问答（进阶） |
| humaneval | `humaneval_gen_0_shot` | 代码生成 |
| mbpp | `mbpp_gen_0_shot` | 代码生成 |
| truthfulqa | `truthful_qa_gen` | 事实性 |
| winogrande | `winogrande_gen` | 常识推理 |
| arc-challenge | `arc_challenge_gen_0_shot` | 科学推理 |

## 配置格式

```yaml
datasets:
  <dataset_key>:              # 自定义 key，与 demand.expectations[].dataset 对应
    config_name: <str>        # AISBench 注册名（上表）
    mode: all                 # 评测模式
    chat_template_kwargs:     # 可选
      thinking: true          # 对于 reasoning 模型
```

## 示例

```yaml
datasets:
  gsm8k:
    config_name: "gsm8k_gen_0_shot_cot_str"
    mode: all
  aime25:
    config_name: "aime2025_gen_0_shot_chat_prompt"
    mode: all
    chat_template_kwargs:
      thinking: true
```
