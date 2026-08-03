# Minimum Fact Ground Truth

该目录只包含两个步骤：数据预处理和最小事实推理。

## 目录

```text
minimum_set/
├── step1_prepare_data.py
├── step2_build_ground_truth.py
├── mrrs/
│   ├── ncbi.py
│   ├── package.py
│   ├── ground_truth.py
│   └── solver.py
├── tests/
├── examples/
├── ground_truth.schema.json
├── requirements.txt
└── data/
    ├── step1/
    └── step2/
```

## Step 1：数据预处理

输入为 `Target_M` Step3 的最终论文：

```powershell
py step1_prepare_data.py `
  --input C:\Users\sxx\Desktop\codex\barchmark-m-7.30\Target_M\data\24_25_3000\step3\final.jsonl
```

每篇 M 生成一个完全本地化的数据包：

```text
data/step1/m1_<PMCID>/
├── manifest.json
├── candidate_reasoning.json
├── fact_coverage.json
├── fact_reference_map.json
├── references_manifest.json
├── target/
│   ├── article.xml
│   ├── introduction.txt
│   └── metadata.json
└── references/
    └── ref1_<PMCID>/
        ├── article.xml
        ├── full_text.txt
        ├── metadata.json
        ├── figures_manifest.json
        └── figures/
```

Step1 只使用 `Target_M` Step3 已通过的 Fact 和参考文献，不重复判断 OA、综述或原始研究类型。

## Step 2：最小 Fact Ground Truth

```powershell
py step2_build_ground_truth.py --model deepseek-v4-pro
```

Step2 固定上游的 Gap 和 Hypothesis，对 2–6 个 Candidate Facts 进行全部组合消融，选择仍能完整定义同一 Gap、支持同一 Hypothesis 的最小 Fact 子集；随后精确求解覆盖这些 Facts 的最小参考文献集合，并执行两层删除验证。

输出：

```text
data/step2/
├── <PMCID>_ground_truth.json
├── run_summary.json
└── cache/
```

最终 JSON 的核心字段：

```text
ground_truth.known_facts
ground_truth.knowledge_gap
ground_truth.hypothesis
minimum_reference_set
two_level_deletion_validation
```

## 测试

```powershell
py -m pytest -q tests
```
