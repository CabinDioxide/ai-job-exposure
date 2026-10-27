"""用 2025 年 11 月美国数据校准工作类比例下限：哪个下限最能复现 Anthropic 公布的美国职业暴露度。"""
import sys, csv, collections, json
import pandas as pd, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, sys.path[0]); import exposure_lib as L
A = sys.argv[1]   # aei 数据目录
onet = L.load_onet(A + '/onet_30_0_Task_Statements.txt', A + '/onet_30_0_Task_Ratings.txt')
beta_key, _ = L.load_beta(A + '/full_labelset.tsv')
share = {}; work = collections.defaultdict(dict); collab = collections.defaultdict(dict)
with open(A + '/v4_claude_ai.csv') as f:
    for r in csv.DictReader(f):
        if r['geo_id'] == 'US' and r['geography'] == 'country' and r['variable'] == 'onet_task_pct':
            share[r['cluster_name'].strip().lower()] = float(r['value'])
        elif r['geography'] == 'global' and r['variable'] == 'onet_task_use_case_pct':
            t, u = r['cluster_name'].rsplit('::', 1); work[t.strip().lower()][u] = float(r['value'])
        elif r['geography'] == 'global' and r['variable'] == 'onet_task_collaboration_pct':
            t, u = r['cluster_name'].rsplit('::', 1); collab[t.strip().lower()][u] = float(r['value'])
share = pd.Series(share)
work_frac = pd.Series({t: d.get('work', 0) / max(1e-9, sum(v for k, v in d.items() if k not in ('not_classified', 'none'))) for t, d in work.items()})
AUTO, AUG = ('directive', 'feedback_loop'), ('task_iteration', 'learning', 'validation')
auto_frac = pd.Series({t: sum(d.get(k, 0) for k in AUTO) / s for t, d in collab.items()
                       if (s := sum(d.get(k, 0) for k in AUTO + AUG)) > 0})
beta = beta_key[beta_key.index.isin(onet.key)]
pub = pd.read_csv(A + '/job_exposure.csv').set_index('occ_code').observed_exposure
pen = pd.read_csv(A + '/task_penetration.csv'); pen['key'] = pen.task.str.strip().str.lower(); pen = pen.groupby('key').penetration.mean()

ceiling = L.occupation_scores(onet, pen, 'key')
k = pub.index.intersection(ceiling.index)
out = {'ceiling_published_task_values_with_proxy_weights': {
    'spearman': round(spearmanr(ceiling[k], pub[k])[0], 3), 'pearson': round(np.corrcoef(ceiling[k], pub[k])[0, 1], 3),
    'mae': round(float((ceiling[k] - pub[k]).abs().mean()), 4)}, 'grid': []}
api = set()
with open(A + '/v4_1p_api.csv') as f:
    for r in csv.DictReader(f):
        if r['geography'] == 'global' and r['facet'] == 'onet_task' and r['variable'] == 'onet_task_count':
            api.add(r['cluster_name'].strip().lower())
for cut in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
  for with_api in (False, True):
    r = L.task_exposure(set(share.index) | (api if with_api else set()), work_frac, auto_frac, beta, cut)
    covered = set(r[r > 0].index); pubcov = set(pen[pen > 0].index)
    occ = L.occupation_scores(onet, r, 'key'); k = pub.index.intersection(occ.index)
    resc = (occ[k] / occ[k].max() - pub[k] / pub[k].max()).abs().mean()
    out['grid'].append({'work_cut': cut, 'with_api': with_api, 'tasks_covered': len(covered), 'mae_rescaled': round(float(resc), 4),
          'task_overlap_with_published': len(covered & pubcov), 'published_covered': len(pubcov),
          'spearman': round(spearmanr(occ[k], pub[k])[0], 3), 'pearson': round(np.corrcoef(occ[k], pub[k])[0, 1], 3),
          'mae': round(float((occ[k] - pub[k]).abs().mean()), 4), 'max_score': round(float(occ.max()), 4)})
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
