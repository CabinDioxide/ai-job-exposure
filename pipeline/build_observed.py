"""用 Anthropic Economic Index 2026-06-26 发布（2026 年 4 月、5 月）重算 12 国实测暴露度。

任务层与职业层公式见 exposure_lib.py（含全球 API 覆盖）；工作类比例下限由 calibrate_us.py 用 2025 年 11 月美国数据校准。
本脚本另外做的几步：
1. 各国每项任务的份额取 4 月、5 月平均，某月未公布按 0 计；全球每项任务的工作类比例、自动化比例同样取两月平均。
2. 任务按原文对到 O*NET 30.0 的职业（同一句任务属于多个职业时，每个职业都算），职业分数按 SOC 2018 六位代码汇总。
   美国仪表盘里的合并代码（如 43-6000）在 O*NET 里没有，取其下各细分职业的平均。
3. 美国按仪表盘职业的 SOC 代码直接取分数。其他国家：SOC 2018 → SOC 2010 取对应代码分数的平均，
   SOC 2010 → ISCO-08 再取平均（分数是强度，不是份额，所以拆分合并都取平均）。
4. 0–10 分：除以本次计算中美国职业的最高分再乘 10，所有国家用同一个除数。

用法：python3 build_observed.py aei数据目录 旧GLOBAL_DATA.json 工作类比例下限 输出目录 [api|no] [每国任务数，0 表示不限]
（api：把全球 API 数据中出现的任务对每国都算作覆盖；no：只用各国 Claude.ai 数据，默认）
"""
import sys, csv, json, collections, pathlib
import pandas as pd, numpy as np
sys.path.insert(0, sys.path[0]); import exposure_lib as L

MONTHS = ['2026-04-01', '2026-05-01']
GEO = {'US': 'USA', 'UK': 'GBR', 'CA': 'CAN', 'AU': 'AUS', 'DE': 'DEU', 'JP': 'JPN', 'KR': 'KOR',
       'FR': 'FRA', 'BR': 'BRA', 'MX': 'MEX', 'IN': 'IND', 'PH': 'PHL'}
SOURCE_TAG = 'Anthropic Economic Index (2026-06 release, Apr–May 2026)'


def load_june(path):
    share = collections.defaultdict(lambda: collections.defaultdict(float))
    glob = collections.defaultdict(lambda: collections.defaultdict(list))
    want = set(GEO.values())
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['category_name'] != 'onet' or r['hierarchy_level'] != '0' or r['date_start'] not in MONTHS:
                continue
            tid = r['node_name'].strip().lower()   # 按任务原文对到职业：同一句任务在 O*NET 里可能属于多个职业，附录把它分给所有这些职业
            if r['geo_level'] == 'country' and r['geo_id'] in want and r['metric_id'] == 'pct':
                share[r['geo_id']][tid] += float(r['value']) / len(MONTHS)
            elif r['geo_level'] == 'global' and r['metric_id'] in ('use_case_work_pct', 'collaboration_bucket_automation_pct', 'pct'):
                glob[r['metric_id']][tid].append(float(r['value']) / 100)
    work = pd.Series({t: np.mean(v) for t, v in glob['use_case_work_pct'].items()})
    auto = pd.Series({t: np.mean(v) for t, v in glob['collaboration_bucket_automation_pct'].items()})
    gshare = pd.Series({t: np.mean(v) * 100 for t, v in glob['pct'].items()})
    return {g: pd.Series(d) for g, d in share.items()}, work, auto, gshare


def load_api(path):
    ids = set()
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['category_name'] == 'onet' and r['hierarchy_level'] == '0' and r['geo_level'] == 'global' \
                    and r['metric_id'] == 'pct' and r['date_start'] in MONTHS:
                ids.add(r['node_name'].strip().lower())
    return ids


def main(A, old_json, gate, outdir, with_api='no', equal_n='0'):
    equal_n = int(equal_n)
    gate = float(gate); out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    onet = L.load_onet(f'{A}/onet_30_0_Task_Statements.txt', f'{A}/onet_30_0_Task_Ratings.txt')
    beta_key, _ = L.load_beta(f'{A}/full_labelset.tsv')
    beta = beta_key.reindex(onet.key.unique()).fillna(0.0)   # Eloundou 未标注的任务按不可行计
    shares, work, auto, gshare = load_june(f'{A}/jun_claude_ai.csv')
    api = load_api(f'{A}/jun_1p_api.csv') if with_api == 'api' else set()

    soc = pd.read_excel(f'{A}/soc_2010_2018.xlsx', header=None, skiprows=9, dtype=str).iloc[:, [0, 2]].dropna()
    soc.columns = ['s10', 's18']; soc = soc.apply(lambda c: c.str.strip())
    to2018 = soc.groupby('s10').s18.apply(list).to_dict()
    x = pd.read_excel(f'{A}/isco_soc.xls', header=None, skiprows=7, dtype=str).iloc[:, [0, 3]].dropna()
    x.columns = ['isco', 's10']; x = x.apply(lambda c: c.str.strip())
    isco2soc = x.groupby('isco').s10.apply(lambda v: sorted(set(v))).to_dict()

    occ = {}; log = {'work_cut': gate, 'equal_n': equal_n, 'global_api_tasks': None, 'tasks_published_per_country': {}, 'tasks_covered_per_country': {}}
    for code, geo in GEO.items():
        pub = shares[geo]
        if equal_n:   # 等样本量：每国只取份额最高的 equal_n 项任务，同份额按全球份额排
            pub = pub.to_frame('s').assign(g=gshare.reindex(pub.index).fillna(0)).sort_values(['s', 'g'], ascending=False).head(equal_n).s
        r = L.task_exposure(set(pub.index) | api, work, auto, beta, gate)
        log['tasks_published_per_country'][code] = len(shares[geo])
        log['tasks_covered_per_country'][code] = int((r > 0).sum())
        occ[code] = L.occupation_scores(onet, r, 'key')
    log['global_api_tasks'] = len(api)
    divisor = float(occ['US'].max()); log['divisor_us_max'] = round(divisor, 4)

    def isco_score(s, isco):
        vals = []
        for s10 in isco2soc.get(isco, []):
            kids = [k for k in to2018.get(s10, [s10]) if k in s.index]
            if kids: vals.append(float(s[kids].mean()))
        return float(np.mean(vals)) if vals else 0.0

    def us_score(s, code):
        # BLS 职业展望里有一些合并代码（如 43-6000、15-1230），O*NET 里没有，取其下各细分职业的平均
        if code in s.index: return float(s[code])
        prefix = code.rstrip('0')
        kids = [k for k in s.index if k.startswith(prefix)]
        return float(s[kids].mean()) if kids else 0.0

    G = json.load(open(old_json))
    for c in G:
        s = occ[c['code']]
        for o in c['occupations']:
            if c['code'] == 'US':
                raw = us_score(s, o['soc_code']); nd = 2
            else:
                raw = isco_score(s, o['isco_code']); nd = 1
            o['observed_exposure_raw'] = round(raw, 4)
            o['observed_exposure'] = round(min(raw / divisor * 10, 10.0), nd)
        c['source'] = c['source'].replace('Anthropic Economic Index V4', SOURCE_TAG).replace('Anthropic V4', SOURCE_TAG)
    json.dump(G, open(out / 'global_data_new.json', 'w'), ensure_ascii=False)
    json.dump(log, open(out / 'build_log.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(log, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:7])
