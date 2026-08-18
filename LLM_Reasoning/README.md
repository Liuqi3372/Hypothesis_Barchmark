# LLM Reasoning Pipeline

本目录实现基准的第三阶段：隐藏 Target M 的真实答案，只向模型提供 Minimum Set 中的最小参考文献证据，让模型独立完成：

```text
参考论文文本与实验图注
        ↓
单篇论文的证据观察与局部结论
        ↓
跨论文证据桥接
        ↓
Knowledge Gap → Hypothesis
        ↓
与密封的 Target M Ground Truth 进行盲评
```

## 运行前准备

在项目根目录进入本文件夹：

```powershell
cd LLM_Reasoning
```

安装运行依赖：

```powershell
py -m pip install openai tqdm
```

设置 DeepSeek API Key：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
```

上游需要先完成：

1. `Target_M`：得到合格目标论文；
2. `Minimum_set/step1_prepare_data.py`：生成论文证据包和 `dataset_manifest.csv`；
3. `Minimum_set/step2_build_ground_truth.py`：生成 `<PMCID>_ground_truth.json`。

## Step 1：导入 Minimum Set 并密封答案

### 代码作用

`import_minimum_set.py` 将 Minimum Set 的输出转换为 LLM Reasoning 使用的统一数据结构。它复制最小参考文献的全文、图注和图片，同时把 Target M 的真实 Knowledge Gap 与 Hypothesis 放入单独的 `sealed_target/`，推理阶段不会读取该答案。

### 单篇论文运行

```powershell
py import_minimum_set.py `
  --package ..\Minimum_set\data\step1\m1_<PMCID> `
  --ground-truth ..\Minimum_set\data\step2\<PMCID>_ground_truth.json `
  --output data\<PMCID>
```

### 批量运行

```powershell
py import_minimum_set.py `
  --source-csv ..\Minimum_set\data\step1\dataset_manifest.csv `
  --package-root ..\Minimum_set\data\step1 `
  --ground-truth-dir ..\Minimum_set\data\step2 `
  --output-root data\benchmark
```

可使用 `--limit 10` 处理前 10 篇，或使用 `--limit 11-20` 处理第 11–20 篇。

### 实现过程

1. 读取 Minimum Set 数据包的 `manifest.json` 和 `references_manifest.json`。
2. 根据 Ground Truth 中的 `minimum_reference_set`，只导入最小参考文献集合。
3. 复制每篇参考论文的 XML、全文、实验图片及图片清单。
4. 为文件计算 SHA-256，保留来源和事实覆盖关系。
5. 将真实 Gap 和 Hypothesis 写入 `sealed_target/target_hypothesis.json`。
6. 生成供后续步骤读取的 `dataset_manifest.json`；批量模式额外生成 `dataset_registry.csv`。

### 输出结构

```text
data/<PMCID>/
├── dataset_manifest.json
├── references/
│   └── ref<N>_<ID>_<PMCID>/
│       ├── source/article.xml
│       ├── source/full_text.txt
│       ├── figures_manifest.json
│       └── figures/
└── sealed_target/
    └── target_hypothesis.json
```

## Step 2：筛选与研究问题相关的真实实验图

### 代码作用

`prepare_reasoning_dataset.py` 对参考论文中的图片进行预处理，只保留包含真实实验数据、并且与论文核心问题相关的图片。机制示意图、流程图、网络图、模拟图、装饰图和不确定图片会被排除。

当前 DeepSeek 流程根据**图注和论文元数据**判断图片类别，不把图片像素发送给模型。原始图片仍保存在本地数据包中。

### 单篇论文运行

```powershell
py prepare_reasoning_dataset.py `
  --data-root data\<PMCID> `
  --model deepseek-v4-pro
```

### 批量运行

```powershell
py prepare_reasoning_dataset.py `
  --source-csv data\benchmark\dataset_registry.csv `
  --model deepseek-v4-pro
```

### 实现过程

1. 读取 `dataset_manifest.json` 及每篇参考论文的 `figures_manifest.json`。
2. 将图注、论文标题和摘要发送给模型进行结构化分类。
3. 只有同时满足“真实实验数据”和“与核心问题相关”的图片才被保留。
4. 将入选图片复制到 `curated/experimental_figures/`。
5. 保存入选、排除及审计记录，保证每个决定都可以追踪。

### 主要输出

```text
data/<PMCID>/
├── curated_dataset_manifest.json
└── references/<reference_id>/curated/
    ├── experimental_figures_manifest.json
    ├── experimental_figures/
    ├── excluded_figures.json
    └── figure_curation_audit.json
```

批量模式还会生成 `curation_registry.csv`。只有状态为 `CURATED` 的数据集会进入 Step 3。

## Step 3：证据驱动的 LLM 推理

### 代码作用

`run_deepseek_reasoning.py` 完成两层推理：先独立分析每篇参考论文，再组合不同论文的局部结论，生成 Knowledge Gap 和可证伪 Hypothesis。

### 单篇论文运行

```powershell
py run_deepseek_reasoning.py `
  --data-root data\<PMCID> `
  --model deepseek-v4-pro
```

如需指定结果目录：

```powershell
py run_deepseek_reasoning.py `
  --data-root data\<PMCID> `
  --output-root data\<PMCID>\reasoning_runs\deepseek_text_v1 `
  --model deepseek-v4-pro
```

### 批量运行

```powershell
py run_deepseek_reasoning.py `
  --source-csv data\benchmark\curation_registry.csv `
  --model deepseek-v4-pro
```

### 第一层：单篇参考文献分析

1. 分别读取每篇参考论文的全文和入选实验图注。
2. 从 Results 和图注中提取实验组、处理条件、测量指标及变化方向。
3. 将证据整理成 `TEXT_DERIVED_FIGURE_EVIDENCE` 观察记录。
4. 为每篇论文生成 2–6 个原子化局部结论，并保留文本锚点和证据 ID。

这一步严格逐篇处理，避免不同论文的证据被提前混合。

### 第二层：跨参考文献推理

1. 每个推理组合至少使用两篇不同参考论文的局部结论。
2. 建立跨论文 evidence bridge，说明不同证据共同确定了什么。
3. 找出任何单篇论文都尚未回答的精确关系，生成 Knowledge Gap。
4. 针对每个 Gap 生成一个方向明确、可以通过实验推翻的 Hypothesis。
5. 校验 Gap 与 Hypothesis 是否一一对应，并检查所有证据 ID 是否真实存在。

### 输出结构

```text
data/<PMCID>/reasoning_runs/deepseek_text_v1/
├── 00_input_snapshot/
├── 01_reference_analysis/
│   └── <reference_id>.json
├── 02_joint_reasoning/
│   └── joint_reasoning.json
├── cache/
└── final_reasoning.json
```

`final_reasoning.json` 是主要结果，包含：

- 每篇参考论文的证据观察和局部结论；
- 跨论文 reasoning combinations；
- Knowledge Gap 集合；
- 与每个 Gap 一一对应的 Hypothesis 集合；
- 模型、输入数据和证据模式等审计信息。

批量模式还会生成 `reasoning_registry.csv`，记录每篇论文的完成状态和最终结果路径。

## 其他文件

- `batch_io.py`：读取和写入批量 CSV，解析 `--limit N` 与 `--limit START-END`。
- `multistage_visual_reasoning_tutorial.ipynb`：用于交互式查看和演示多阶段推理流程。
- `replace_review_reference.py`：针对特定测试数据替换综述参考文献的维护脚本，不是标准流水线的必要步骤。

## 缓存和重新运行

模型响应会写入 `cache/`。默认情况下，重复运行会复用缓存，适合断点续跑。需要忽略旧缓存并重新调用模型时，在 Step 2 或 Step 3 命令后添加：

```powershell
--force
```

## 最简批量运行顺序

```powershell
py import_minimum_set.py `
  --source-csv ..\Minimum_set\data\step1\dataset_manifest.csv `
  --package-root ..\Minimum_set\data\step1 `
  --ground-truth-dir ..\Minimum_set\data\step2 `
  --output-root data\benchmark

py prepare_reasoning_dataset.py `
  --source-csv data\benchmark\dataset_registry.csv `
  --model deepseek-v4-pro

py run_deepseek_reasoning.py `
  --source-csv data\benchmark\curation_registry.csv `
  --model deepseek-v4-pro
```
