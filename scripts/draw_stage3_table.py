"""分支函数逻辑表"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Microsoft YaHei'

WHITE = 'white'
BLACK = 'black'
GRAY = '#F0F0F0'
DARK = '#444444'


def box(ax, x, y, w, h, text, bold=False, bg=WHITE, edge=BLACK, lw=1.5, white_text=False):
    lines = text.split('\n')
    n = len(lines)
    fs = (h / (n + 3)) * 72 * 0.65
    weight = 'bold' if bold else 'normal'
    color = 'white' if white_text else BLACK
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=bg, edgecolor=edge, linewidth=lw)
    ax.add_patch(b)
    line_h = h / (n + 3)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - line_h*(i+2), line,
                ha='center', va='center', fontsize=fs,
                fontweight=weight, color=color)


def make():
    fig, ax = plt.subplots(1, 1, figsize=(44, 28))
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 28)
    ax.axis('off')
    ax.set_facecolor('white')

    ax.text(22, 27.2, 'Stage 3 第二层 —— 8个分支函数内部逻辑', ha='center', fontsize=32, fontweight='bold')
    ax.text(22, 26.2, '各分支只接收自己需要的参数，内部完成"数据结构特征 → 具体估计量"的选择',
            ha='center', fontsize=17, color=DARK)

    col_w = [2.5, 4.0, 5.5, 8.5, 6.0, 8.5]
    labels = ['分支', '函数名', '输入参数', '内部判断逻辑', '输出方法', '数据兼容性警告']
    col_x = [1.5]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w + 0.3)

    header_y = 25.0
    for cx, w, lbl in zip(col_x, col_w, labels):
        box(ax, cx, header_y, w, 0.9, lbl, bold=True, bg=BLACK, white_text=True)

    rows = [
        ['A', '_branch_random\n_assignment', '无', '无内部判断', '随机化推断\n(Fisher exact test)', '无'],
        ['B', '_branch_threshold', 'threshold_type', 'sharp → 精确RDD\nfuzzy → 模糊RDD\n未指定 → 返回警告', '精确RDD\n或 模糊RDD(IV)', 'threshold_type\n未指定时警告'],
        ['C', '_branch_selection\n_on_observables', 'high_dimensional\n_controls\npanel_available', '控制变量多 → DML\n控制变量少 → PSM/IPW/AIPW\n有面板数据 → 额外警告', 'DML (高维时)\n或 PSM/IPW/AIPW', 'panel_available=True:\n"有面板为何不用DID?"'],
        ['D', '_branch_single\n_policy_shock', 'panel_available\nhas_control_group', '有对照组 → 标准DID(TWFE)\n无对照组 → SCM', '标准DID(TWFE)\n或 SCM', 'panel_available=False:\n"DID需要面板数据"'],
        ["D'", '_branch_staggered\n_policy_shock', 'panel_available\nhas_control_group\neveryone_treated\n_eventually', '有从未处理组\n  → C&S never-treated\n全部最终被处理\n  → C&S not-yet-treated', 'C&S(2021)\nnever-treated\n或 not-yet-treated', 'panel_available=False:\n"交错DID需要面板"'],
        ['E', '_branch_time_varying\n_unobservables', 'has_instrument\npanel_available', '有工具变量 → 2SLS/LIML\n无工具变量 → SCM/IFE', '2SLS / LIML\n或 SCM / IFE', '无面板且无工具:\n"因果效应可能不可识别"'],
        ['F', '_branch_continuous\n_intensity', 'panel_available', '有面板 → 强度DID\n无面板 → 警告', '强度DID\n(Continuous DiD)', 'panel_available=False:\n"截面强度变异\n几乎总是内生的"'],
        ['G', '_branch_multiple\n_policies', '无', '无内部判断', '三重差分(DDD)\n或 控制回归', '无'],
    ]

    row_h = 3.0
    start_y = 23.5

    for i, row in enumerate(rows):
        y = start_y - i * row_h
        bg = WHITE if i % 2 == 0 else GRAY
        for j, (cell, cx) in enumerate(zip(row, col_x)):
            box(ax, cx, y-row_h+0.2, col_w[j], row_h-0.15, str(cell), bg=bg)

    ax.text(22, start_y - len(rows)*row_h - 1.0,
            '所有分支函数返回前统一附加:  (1) 异质性分析建议 (Causal Forest)    (2) 敏感性分析建议 (sensitivity_analysis.py)',
            ha='center', fontsize=15, color=DARK)

    plt.tight_layout(pad=2)
    outpath = r'C:\Users\徐铭洲\Desktop\stage3_分支函数逻辑表.png'
    plt.savefig(outpath, dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'OK: {outpath}')


if __name__ == '__main__':
    make()
