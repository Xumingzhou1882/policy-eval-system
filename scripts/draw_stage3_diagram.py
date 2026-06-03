"""Stage 3 双层决策树结构图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.family'] = 'Microsoft YaHei'

WHITE = 'white'
BLACK = 'black'
GRAY = '#F0F0F0'
DARK = '#444444'


def box(ax, x, y, w, h, text, bold=False, bg=WHITE, edge=BLACK, lw=1.5):
    """画框+文字。字号自动适配框高和行数，留足边距保证不溢出。"""
    lines = text.split('\n')
    n = len(lines)
    # 关键: n+3 保证上下有内边距, *0.65 补偿matplotlib的行高/字号比
    fs = (h / (n + 3)) * 72 * 0.65

    weight = 'bold' if bold else 'normal'
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                        facecolor=bg, edgecolor=edge, linewidth=lw)
    ax.add_patch(b)

    line_h = h / (n + 3)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - line_h*(i+2), line,
                ha='center', va='center', fontsize=fs,
                fontweight=weight, color=BLACK)


def arrow(ax, x1, y1, x2, y2, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=BLACK, lw=lw))


def make():
    fig, ax = plt.subplots(1, 1, figsize=(38, 44))
    ax.set_xlim(0, 38)
    ax.set_ylim(0, 44)
    ax.axis('off')
    ax.set_facecolor('white')

    # ═══ 标题 ═══
    ax.text(19, 43.0, 'Stage 3 双层决策树 —— 完整结构图', ha='center', fontsize=30, fontweight='bold')
    ax.text(19, 42.0, 'classify_mechanism() 事实→机制     │     decide_method() 机制→方法',
            ha='center', fontsize=16, color=DARK)

    # ═══ Stage 2 输入 ═══
    box(ax, 4, 40.0, 30, 1.2,
        'Stage 2 结构化事实 JSON:  q1分配 | q2阈值 | q3时间 | q4对照 | q5类型 | q6多政策 | q7工具',
        bold=True, bg=GRAY)
    arrow(ax, 19, 40.0, 19, 39.0)

    # ═══ 第一层 ═══
    box(ax, 3, 37.0, 32, 1.5,
        '第一层: classify_mechanism()  ——  7条优先级规则，命中即停',
        bold=True, bg=GRAY)
    arrow(ax, 19, 37.0, 19, 35.2)

    # 7条规则 —— 第一排5条
    rules_h = 4.5
    rules_w = 5.6
    gap = 0.5
    r1_texts = [
        '规则1: 随机分配\n\n处理是抽签决定的?\n是 → 随机化推断\n否 → 继续',
        '规则2: 阈值规则\n\n是否有分数线/门槛值?\n是 → RDD(精确/模糊)\n否 → 继续',
        '规则3: 无时间维度\n\n政策有已知开始时间?\n否 → 可观测选择\n是 → 继续',
        '规则4: 多政策叠加\n\n同期是否有多个政策?\n是 → DDD\n否 → 继续',
        '规则5: 连续强度\n\n处理是否有剂量差异?\n是 → 强度DID\n否 → 继续',
    ]
    r1_x = [0.3 + i*(rules_w+gap) for i in range(5)]
    r1_y = 30.2
    for x, text in zip(r1_x, r1_texts):
        box(ax, x, r1_y, rules_w, rules_h, text)
    for i in range(4):
        arrow(ax, r1_x[i]+rules_w, r1_y+rules_h/2, r1_x[i+1], r1_y+rules_h/2)

    # 7条规则 —— 第二排
    r2_y = 24.0
    rules_h2 = 5.5
    r2_texts = [
        '规则6: 分批推进\n\n处理是否分批开始?\n是 → 交错政策冲击\n否 → 继续',
        '规则7: 单次冲击\n\n(规则6的否分支)\n→ 单次政策冲击',
        '输出\n\nmechanism\n+\n8个flag',
        '传给第二层\n\ndecide_method()\n根据mechanism\n路由到8个分支函数',
        '补充说明\n\ntime_varying_\nunobservables\n不从事实推断\n需手动指定机制',
    ]
    r2_x = r1_x
    for x, text in zip(r2_x, r2_texts):
        box(ax, x, r2_y, rules_w, rules_h2, text)
    arrow(ax, r1_x[4]+rules_w/2, r1_y, r2_x[0]+rules_w/2, r2_y+rules_h2+0.2)
    for i in range(4):
        arrow(ax, r2_x[i]+rules_w, r2_y+rules_h2/2, r2_x[i+1], r2_y+rules_h2/2)

    # ═══ 第一层 → 第二层 ═══
    box(ax, 4, 21.5, 30, 1.2,
        '第一层输出 → mechanism + 8个flag → 传入第二层',
        bold=True, bg=GRAY)
    arrow(ax, 19, 21.5, 19, 20.3)

    # ═══ 第二层 ═══
    box(ax, 3, 18.0, 32, 1.5,
        '第二层: decide_method()  根据mechanism路由到8个分支函数，各分支只收自己参数',
        bold=True, bg=GRAY)
    arrow(ax, 19, 18.0, 19, 16.0)

    # 8个分支函数
    bw = 4.2
    bgap = 0.3
    bh = 5.5
    by = 10.0
    bx = [0.2 + i*(bw+bgap) for i in range(8)]
    b_texts = [
        '分支A\n随机分配\n─────────\n参数: 无\n→ 随机化推断',
        '分支B\n阈值规则\n─────────\n参数: thresh.\n sharp→精确\n fuzzy→模糊',
        '分支C\n可观测选择\n─────────\n参数: high_dim\n     panel\n高维→DML\n低维→PSM/IPW',
        '分支D\n单次政策冲击\n─────────\n参数: panel\n     has_ctrl\n有对照→DID\n无对照→SCM',
        "分支D'\n交错政策冲击\n─────────\n参数: panel, ctrl\n     everyone\n从未→C&S never\n全处理→not-yet",
        '分支E\n时变不可观测\n─────────\n参数: has_iv\n     panel\n有IV→2SLS\n无IV→SCM/IFE',
        '分支F\n连续处理强度\n─────────\n参数: panel\n有面板→强度DID\n无面板→警告',
        '分支G\n多政策叠加\n─────────\n参数: 无\n→ 三重差分DDD',
    ]
    for x, text in zip(bx, b_texts):
        box(ax, x, by, bw, bh, text, edge=BLACK, lw=1)

    # ═══ 统一输出 ═══
    arrow(ax, 19, 10.0, 19, 8.0, lw=2)

    box(ax, 5, 4.5, 28, 3.5, '', edge=BLACK, lw=2)
    ax.text(19, 7.5, '统一的 MethodRecommendation', ha='center', fontsize=24, fontweight='bold')
    ax.text(19, 6.8, '主推荐方法  |  识别变异来源  |  核心假设(标注:可检验/需论证)  |  回退方案(2-3个)', ha='center', fontsize=15, color=DARK)
    ax.text(19, 6.1, '必需变量  |  可选变量  |  关键文献  |  数据兼容性警告  |  异质性分析建议(Causal Forest)', ha='center', fontsize=15, color=DARK)

    box(ax, 5, 2.0, 28, 1.8,
        '调用方式\n(A) --from-facts stage2_facts.json   →  运行 Level 1 + Level 2        (B) --mechanism X --has-control-group  →  仅 Level 2',
        bold=False, bg=GRAY)

    plt.tight_layout(pad=1.5)
    outpath = r'C:\Users\徐铭洲\Desktop\stage3_决策树结构图.png'
    plt.savefig(outpath, dpi=120, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'OK: {outpath}')


if __name__ == '__main__':
    make()
