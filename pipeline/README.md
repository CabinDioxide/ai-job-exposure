# 实测 AI 暴露度的计算脚本 / Observed-exposure pipeline

方法与限制见 `methodology.html` 第一节与第二节第 7–9 条。See `methodology.html` for the method and its limits.

需要的数据（放在同一个目录里，下称 `aei/`）/ Inputs, in one folder:

| 文件 | 来源 |
|---|---|
| `jun_claude_ai.csv` | Hugging Face `Anthropic/EconomicIndex`，`release_2026_06_26/data/aei_claude_ai_2026-06-26.csv` |
| `jun_1p_api.csv` | 同上，`aei_1p_api_2026-06-26.csv`（只在打开 API 开关时用） |
| `v4_claude_ai.csv`、`v4_1p_api.csv` | 同上，`release_2026_01_15/data/intermediate/`（校准用） |
| `job_exposure.csv`、`task_penetration.csv` | 同上，`labor_market_impacts/`（校准用） |
| `onet_30_0_Task_Statements.txt`、`onet_30_0_Task_Ratings.txt` | O*NET 30.0 数据库 |
| `full_labelset.tsv` | github.com/openai/GPTs-are-GPTs `data/` |
| `soc_2010_2018.xlsx`、`isco_soc.xls` | BLS SOC 2010–2018 与 ISCO-08–SOC 2010 交叉表 |

```bash
python3 pipeline/calibrate_us.py aei pipeline/results/calibration.json
python3 pipeline/build_observed.py aei 旧GLOBAL_DATA.json 0.1 输出目录 no 0
```

`build_observed.py` 输出 `global_data_new.json`，替换 `index.html` 中 `const GLOBAL_DATA = ` 后面的数组即可。2026 年 10 月这一版用的参数：工作类比例下限 0.1，不计 API，不限任务数（`results/build_log.json`）。
