"""按 Anthropic《Labor market impacts of AI》附录（2026-03）的公式，计算分国家的实测暴露度。

任务层：r_t = 1[该国 Claude.ai 数据或全球 API 数据中出现] × 1[全球工作类比例 ≥ 下限] × 1[β_t > 0] × α_t，α_t = 0.5 + 0.5 × 自动化占比_t。
职业层：职业内各任务 r_t 的加权平均，权重为任务所占工作时间。

与 Anthropic 原做法的差别（数据所限）：
1. 国家数据只公布过了样本门槛的任务，所以「使用量足够」改为「任务出现在该国数据里」；
   另设全球工作类比例下限，用美国校准。
2. 国家数据没有分用途的份额，工作类比例用全球同期每项任务的数值。
3. API 流量只有全球数据，没有分国家的。可选：把全球 API 数据中出现的任务对每国都算作覆盖（build_observed.py 的 api 开关）；
   美国校准显示这样更接近 Anthropic 公布值，但 API 部分各国相同，会抹平国家差别。
   自动化占比用全球同期 Claude.ai 每项任务的自动化比例。
4. Anthropic 的任务时间份额（Tamkin & McCrory 2025，由 Claude 估计）没有公开，
   用 O*NET 任务评分里的频率（FT）折算为每年次数作权重，校准时报告这一替代造成的误差。
"""
import pandas as pd, numpy as np

FT_PER_YEAR = {1: 0.5, 2: 4, 3: 12, 4: 52, 5: 260, 6: 520, 7: 2080}


def load_onet(statements, ratings):
    ts = pd.read_csv(statements, sep='\t')
    ts['key'] = ts.Task.str.strip().str.lower()
    ts['soc6'] = ts['O*NET-SOC Code'].str[:7]
    tr = pd.read_csv(ratings, sep='\t')
    ft = tr[tr['Scale ID'] == 'FT']
    ft = ft.assign(w=ft.Category.map(FT_PER_YEAR) * ft['Data Value'] / 100)
    w = ft.groupby(['O*NET-SOC Code', 'Task ID']).w.sum().rename('w').reset_index()
    ts = ts.merge(w, on=['O*NET-SOC Code', 'Task ID'], how='left')
    ts['w'] = ts.w.fillna(ts.w.median())   # 少数任务没有频率评分，取中位数
    return ts[['O*NET-SOC Code', 'soc6', 'Task ID', 'key', 'w']]


def load_beta(path):
    b = pd.read_csv(path, sep='\t')
    b['key'] = b.Task.str.strip().str.lower()
    return b.groupby('key').beta.max(), b.groupby('Task ID').beta.max()


def task_exposure(published, work_frac, auto_frac, beta, work_cut):
    """published：该国数据里出现过的任务编号集合（出现即已过 Anthropic 的样本门槛；
    6 月数据的份额四舍五入到两位小数，小份额显示为 0，所以按是否出现判断，不按数值）。
    work_frac、auto_frac、beta 以任务为索引；work_cut：全球工作类比例的下限，用美国校准。"""
    idx = beta.index
    pub = pd.Series(idx.isin(list(published)), index=idx)
    wf = work_frac.reindex(idx).fillna(work_frac.median())
    a = 0.5 + 0.5 * auto_frac.reindex(idx).fillna(0.0)   # 没有自动化数据的任务按只有辅助型使用计（附录：α=0.5）
    return (pub & (wf >= work_cut) & (beta > 0)).astype(float) * a


def occupation_scores(tasks, r, keycol):
    t = tasks.assign(r=tasks[keycol].map(r).fillna(0.0))
    return t.groupby('soc6').apply(lambda g: np.average(g.r, weights=g.w), include_groups=False)
