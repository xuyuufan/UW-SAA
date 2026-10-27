"""Export complete GP-only MAIN figures and a concise reproducible report."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

DEFAULT_RESULTS = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir', type=Path, default=DEFAULT_RESULTS)
parser.add_argument('--output-dir', type=Path, default=DEFAULT_RESULTS)
parser.add_argument('--font', default='Noto Sans CJK JP', help='installed font family supporting Chinese')
args = parser.parse_args()
OUT = args.output_dir.resolve()
DATA = args.data_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
try:
    font = FontProperties(fname=findfont(FontProperties(family=args.font), fallback_to_default=False))
    bold = FontProperties(fname=findfont(FontProperties(family=args.font, weight='bold'), fallback_to_default=False))
except ValueError as exc:
    raise SystemExit('Install Noto Sans CJK (fonts-noto-cjk on Debian/Ubuntu), or pass --font with an installed Chinese font.') from exc
plt.rcParams.update({'font.family': font.get_name(), 'font.size': 11,
    'axes.unicode_minus': False, 'pdf.fonttype': 42, 'figure.facecolor': '#F5F8FB',
    'savefig.facecolor': '#F5F8FB', 'axes.edgecolor': '#D3DDE5', 'text.color': '#1E3349',
    'axes.labelcolor': '#41556A', 'xtick.color': '#667A8E', 'ytick.color': '#263C52'})
NAVY, GREEN, RUST, GRAY = '#22496F', '#087F72', '#B35D3F', '#718497'

def read_csv(name):
    with (DATA / name).open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ('relative_reduction', 'ci95_lower', 'ci95_upper', 'baseline_macro_mean', 'candidate_macro_mean'):
            if row.get(key):
                row[key] = float(row[key])
    return rows

families = read_csv('family_effects.csv')
conditions = read_csv('condition_effects.csv')
f = {r['family']: r for r in families if r['metric'] == 'relocations'}
primary = [r for r in conditions if r['metric'] == 'relocations']
overall = f['SS_TW_DP_equal_family_mean']
clean = next(r for r in primary if r['family'] == 'CLEAN')
ordered = [overall, f['subset_shuffle'], f['time_window_shuffle'], f['delayed_positions'], clean]
labels = ['三类噪声等权 · 主结果', 'SS 子集打乱', 'TW 时间窗口打乱', 'DP 延迟位置', 'CLEAN 无噪声']

def pct(x):
    return f'{x*100:.2f}%'

def axes_style(ax):
    ax.set_facecolor('white')
    ax.spines[['top', 'left', 'right']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.tick_params(axis='x', length=3, labelsize=10)
    ax.grid(axis='x', color='#E2EAF1', linewidth=.65)
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(14.5, 8.6))
fig.text(.045, .944, 'GP 在 Caserta 上的完整配对测试结果', fontproperties=bold, fontsize=25)
fig.text(.045, .898, 'UWSAA 对比 NATIVE20  ·  本地与云端结果已合并  ·  主指标：每条轨迹的搬移次数',
         fontproperties=font, fontsize=12, color='#63798C')
cards = [(.045, .30, '三类噪声等权平均', f'搬移减少 {pct(overall["relative_reduction"])}',
          f'95% CI  {pct(overall["ci95_lower"])} – {pct(overall["ci95_upper"])}', GREEN),
         (.365, .30, '无噪声条件的代价', f'搬移增加 {pct(-clean["relative_reduction"])}',
          f'95% CI  {pct(-clean["ci95_upper"])} – {pct(-clean["ci95_lower"])}', RUST),
         (.685, .27, '完整性与配对校验', '68,880 条全部通过', '41 条件 × 840 布局 × 2 方法', NAVY)]
for x, w, title, number, sub, color in cards:
    fig.add_artist(FancyBboxPatch((x, .755), w, .105, boxstyle='round,pad=0.008,rounding_size=.009',
        transform=fig.transFigure, facecolor='white', edgecolor='#E0E9F1', linewidth=.8))
    fig.text(x+.012, .833, title, fontproperties=font, fontsize=10.5, color='#637B90')
    fig.text(x+.012, .798, number, fontproperties=bold, fontsize=19, color=color)
    fig.text(x+.012, .772, sub, fontproperties=font, fontsize=10, color='#6A8093')

grid = fig.add_gridspec(1, 2, left=.20, right=.94, bottom=.24, top=.665, wspace=.48,
                       width_ratios=[1.1, .9])
ax = fig.add_subplot(grid[0, 0])
axes_style(ax)
ax.axhspan(-.45, .45, color='#E5F0F7', zorder=0)
ax.axhspan(3.55, 4.45, color='#FCF0EA', zorder=0)
for i, row in enumerate(ordered):
    point, low, high = [row[k]*100 for k in ('relative_reduction', 'ci95_lower', 'ci95_upper')]
    color = NAVY if i == 0 else (GREEN if point >= 0 else RUST)
    ax.plot([low, high], [i, i], color=color, linewidth=2.2, solid_capstyle='round', zorder=3)
    ax.scatter([point], [i], marker='D' if i == 0 else 'o', s=64 if i == 0 else 42, color=color, zorder=4)
    ax.text(1.025, i, f'{point:+.2f}%', transform=ax.get_yaxis_transform(), va='center',
            fontsize=11.5, color=color, fontweight='bold' if i == 0 else 'normal')
ax.set_yticks(range(5), labels)
ax.set_ylim(4.55, -.65)
ax.set_xlim(-3.3, 13.5)
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}%'))
ax.axvline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=1)
ax.set_xlabel('相对搬移减少率 · 正值有利于 UWSAA', fontproperties=font, labelpad=12, fontsize=10.5)
ax.set_title('A   主要效果与 95% 置信区间', fontproperties=bold, loc='left', pad=21, fontsize=13)

bar = fig.add_subplot(grid[0, 1])
axes_style(bar)
bar_rows = [f['subset_shuffle'], f['time_window_shuffle'], f['delayed_positions'], clean]
ys = np.arange(4)
baseline = [r['baseline_macro_mean'] for r in bar_rows]
candidate = [r['candidate_macro_mean'] for r in bar_rows]
bar.barh(ys-.18, baseline, height=.30, color=NAVY, label='NATIVE20', zorder=3)
bar.barh(ys+.18, candidate, height=.30, color=[GREEN, GREEN, GREEN, RUST], label='UWSAA', zorder=3)
for i, (b, c) in enumerate(zip(baseline, candidate)):
    bar.text(b+.45, i-.18, f'{b:.2f}', va='center', color=NAVY, fontsize=10.5)
    bar.text(c+.45, i+.18, f'{c:.2f}', va='center', color=GREEN if i < 3 else RUST, fontsize=10.5)
bar.set_yticks(ys, ['SS', 'TW', 'DP', 'CLEAN'])
bar.set_ylim(3.65, -.8)
bar.set_xlim(0, max(baseline+candidate)*1.15)
bar.set_xlabel('宏平均搬移次数 · 越少越好', fontproperties=font, labelpad=12, fontsize=10.5)
bar.set_title('B   每条轨迹的平均搬移次数', fontproperties=bold, loc='left', pad=21, fontsize=13)
bar.legend(frameon=False, loc='upper left', bbox_to_anchor=(-.04, 1.035), ncol=2, fontsize=9.5,
           handlelength=1.2, columnspacing=1.4)

fig.text(.045, .143, '主结果按 SS、TW、DP 三个家族的相对减少率等权平均；Overlay、Regime 和 CLEAN 分开报告。',
         fontproperties=font, fontsize=10.2, color='#435D73')
fig.text(.045, .111, '统计：21 种几何形状等权；10,000 次配对布局簇 bootstrap（PCG64）。区间为逐项 95% CI。',
         fontproperties=font, fontsize=10, color='#718598')
fig.text(.045, .079, '完整 GP-only MAIN  ·  本地 23,520 条 + 云端 45,360 条  ·  独立统计实现核验通过  ·  不代表六求解器或 MEAN6',
         fontproperties=font, fontsize=9.8, color='#718598')
fig.savefig(OUT / 'gp_caserta_summary.png', dpi=170)
fig.savefig(OUT / 'gp_caserta_summary.pdf', metadata={'CreationDate': None, 'ModDate': None})
plt.close(fig)

def condition_label(row):
    cid = row['condition_id']
    if row['family'] == 'CLEAN':
        return 'CLEAN'
    repeat = int(row['repeat'])
    if row['family'] == 'subset_shuffle':
        return f'SS p={cid.split("p=")[1].split("|")[0]} · r{repeat}'
    if row['family'] == 'time_window_shuffle':
        return f'TW B={cid.split("B=")[1].split("|")[0]} · r{repeat}'
    if row['family'] == 'delayed_positions':
        return f'DP p={cid.split("p=")[1].split("|")[0]} · r{repeat}'
    if row['family'] == 'overlay_mix':
        return f'Overlay · r{repeat}'
    return f'{cid.split("profile=")[1].split("|")[0]} · r{repeat}'

manifest = json.loads((DATA / 'analysis_manifest.json').read_text(encoding='utf-8'))
registered_ids = [condition for item in manifest['inputs'] for condition in item['conditions']]
order = {condition: index for index, condition in enumerate(registered_ids)}
if len(order) != 41 or {row['condition_id'] for row in primary} != set(order):
    raise ValueError('Figure inputs must cover the same 41 conditions as the analysis manifest')
primary.sort(key=lambda r: order[r['condition_id']])
fig, axes = plt.subplots(1, 2, figsize=(15.5, 12.5))
fig.subplots_adjust(left=.16, right=.91, top=.85, bottom=.14, wspace=.72)
fig.text(.045, .947, '41 个条件的完整 GP 配对效果', fontproperties=bold, fontsize=25)
fig.text(.045, .906, 'UWSAA 对比 NATIVE20  ·  搬移次数减少率：正值更好，负值更差  ·  68,880 条完整记录',
         fontproperties=font, fontsize=12, color='#627A8E')
for axis, selected, title in zip(axes, (primary[:20], primary[20:]),
        ('A   纯噪声条件：SS / TW / DP', 'B   Overlay / Regime / CLEAN')):
    axes_style(axis)
    for i, row in enumerate(selected):
        if (i//2) % 2 == 0:
            axis.axhspan(i-.48, i+.48, color='#EBF2F7', zorder=0)
        point, low, high = [row[k]*100 for k in ('relative_reduction', 'ci95_lower', 'ci95_upper')]
        color = GREEN if point >= 0 else RUST
        axis.plot([low, high], [i, i], color=color, linewidth=1.8, solid_capstyle='round', zorder=3)
        axis.scatter(point, i, color=color, s=29, zorder=4)
        axis.text(1.035, i, f'{point:+.2f}%', transform=axis.get_yaxis_transform(), va='center', color=color, fontsize=10)
    lo = min(0, *(r['ci95_lower']*100 for r in selected))
    hi = max(0, *(r['ci95_upper']*100 for r in selected))
    pad = max((hi-lo)*.08, .4)
    axis.set_xlim(lo-pad, hi+pad)
    axis.set_ylim(len(selected)-.45, -.75)
    axis.set_yticks(range(len(selected)), [condition_label(r) for r in selected])
    axis.tick_params(axis='y', labelsize=10.2)
    axis.axvline(0, color=GRAY, linestyle=(0, (3, 3)), linewidth=1)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}%'))
    axis.set_xlabel('UWSAA 相对搬移减少率', fontproperties=font, labelpad=10, fontsize=10.5)
    axis.set_title(title, fontproperties=bold, loc='left', pad=20, fontsize=13)
fig.text(.045, .090, '点为估计值，线为逐项 95% 置信区间；r0/r1 为注册重复。21 种几何形状等权，布局内配对。',
         fontproperties=font, fontsize=10.2, color='#435D73')
fig.text(.045, .060, '10,000 次 PCG64 布局簇 bootstrap；区间未作同时多重比较校正。GP-only，独立统计实现核验通过。',
         fontproperties=font, fontsize=10, color='#718598')
fig.savefig(OUT / 'gp_caserta_all_conditions.png', dpi=170)
fig.savefig(OUT / 'gp_caserta_all_conditions.pdf', metadata={'CreationDate': None, 'ModDate': None})
plt.close(fig)

print(json.dumps({'status': 'PASS', 'figures': ['gp_caserta_summary.png', 'gp_caserta_summary.pdf',
    'gp_caserta_all_conditions.png', 'gp_caserta_all_conditions.pdf'], 'data_dir': str(DATA), 'output_dir': str(OUT)}))
