# Target M Pipeline

三个入口按顺序运行：Step1 收集、Step2 科学质量筛选、Step3 证据链可行性筛选。

```text
Target_M/
├── step1_collect_pmc.py
├── step2_llm_screen.py
├── step3_reference_feasibility.py
├── pmc_m/
├── tests/
└── data/<dataset>/
    ├── step1/
    ├── step2/
    └── step3/
```

## Step1

```powershell
py step1_collect_pmc.py `
  --year-range 2024-2025 `
  --count 3000 `
  --output C:\Users\sxx\Desktop\codex\barchmark-m-7.30\Target_M\data\24_25_3000
```

3000 篇候选保存在 `step1/eligible.csv`。

## Step2

```powershell
py step2_llm_screen.py `
  --provider deepseek `
  --model deepseek-v4-pro `
  --input C:\Users\sxx\Desktop\codex\barchmark-m-7.30\Target_M\data\24_25_3000
```

## Step3：直接扫描3000篇CSV，找到指定数量

```powershell
py step3_reference_feasibility.py `
  --source-csv C:\Users\sxx\Desktop\codex\barchmark-m-7.30\Target_M\data\24_25_3000\step1\eligible.csv `
  --target-count 5 `
  --model deepseek-v4-pro `
  --input C:\Users\sxx\Desktop\codex\barchmark-m-7.30\Target_M\data\24_25_3000
```

`--target-count 5` 表示持续扫描 CSV，直到累计得到 5 篇 Step3 合格论文。终端同时显示：

- `Scanned CSV papers`：扫描进度；
- `Qualified papers`：合格论文数量。

已存在于 `step3/final.jsonl` 或 `step3/excluded.jsonl` 的论文默认跳过，因此命令可以安全续跑。使用 `--force` 才会重新扫描。

输出：

```text
step3/
├── final.jsonl
├── final.csv
├── excluded.jsonl
├── excluded.csv
├── summary.json
├── reasoning_inputs/<PMCID>/feasibility_audit.json
└── cache/
```

所有 API Key 从项目根目录读取：

```text
C:\Users\sxx\Desktop\codex\barchmark-m-7.30\key
```

## 测试

```powershell
py -m pytest -q tests
```
