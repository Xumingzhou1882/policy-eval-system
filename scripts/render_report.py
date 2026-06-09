"""Academic report renderer: Markdown + XeLaTeX from shared content model.

Generates a complete academic paper draft:
  Abstract → Introduction → Institution & Theory → Research Design →
  Empirical Results → Robustness → Conclusion → References → Appendix
"""
import argparse, json, math, re
from datetime import datetime
from pathlib import Path


# ═══════════════════════════ Formatters ═══════════════════════════
def _s(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def _normal_pval(coef_val, se_val):
    """Compute two-sided p-value from coefficient and SE using normal approximation."""
    if se_val <= 0:
        return 1.0
    t_stat = abs(coef_val / se_val)
    # Use math.erf for normal CDF
    return max(2 * (1 - 0.5 * (1 + math.erf(t_stat / math.sqrt(2)))), 1e-300)


def _coef(c, p):
    v = float(c)
    if abs(v) < 0.0001 and v != 0:
        fmt = f"{v:.2e}"
    elif abs(v) >= 1000:
        fmt = f"{v:.2f}"
    else:
        fmt = f"{v:.4f}"
    return fmt + _s(p) if p is not None else fmt


def _se(v):
    val = float(v)
    if abs(val) < 0.0001 and val != 0:
        fmt = f"({val:.2e})"
    elif abs(val) >= 1000:
        fmt = f"({val:.2f})"
    else:
        fmt = f"({val:.4f})"
    return fmt


def _p(p):
    return "<0.001" if (p or 1) < 0.001 else f"{p:.4f}"


def _sig(p):
    return "1%" if p < 0.01 else ("5%" if p < 0.05 else ("10%" if p < 0.1 else "not sig"))


def _slug(t):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in t).lower()


def _find_or_create_subject_dir(policy_name: str, base_dir):
    """Find an existing subject folder or create a new one.

    Avoids creating duplicate folders for the same policy entered with
    different spellings (e.g., 'LTCI' vs '长期护理保险试点').
    """
    from pathlib import Path
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(policy_name)

    existing = [d for d in base_dir.iterdir() if d.is_dir()]

    # 1. Exact match
    for d in existing:
        if d.name.lower() == slug:
            return d

    # 2. Slug is a substring of an existing dir (or vice versa)
    for d in existing:
        if slug in d.name.lower() or d.name.lower() in slug:
            return d

    # 3. First 5 chars match
    for d in existing:
        if len(d.name) >= 5 and len(slug) >= 5:
            if d.name[:5].lower() == slug[:5]:
                return d

    # 4. No match — create new
    new_dir = base_dir / slug
    new_dir.mkdir(parents=True, exist_ok=True)
    return new_dir


def _safe_sort_key(k):
    """Sort event study period keys safely (handle non-integer keys like 'avg', 'post_avg')."""
    try:
        return (0, float(k))
    except (ValueError, TypeError):
        return (1, str(k))


# Chinese section numbers
_SN = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]


# ═══════════════════════════ Shared content builder ═══════════════════════════
def _build_report(data):
    """Build a list of section dicts. Both MD and LaTeX renderers consume this."""
    O = data.get
    sec = O("sections", {})
    chain = O("method_chain", {})
    main = O("main_result", {})
    model = O("model_spec", {})
    meta = O("data_meta", {})
    appendix = O("appendix", {})
    desc = O("descriptive_stats", {})
    es = O("event_study", {})
    policy = O("policy", "该政策")
    outcome = O("outcome", "结果变量")

    coef = main.get("coefficient", 0)
    se_val = main.get("std_error", 0)
    pval = main.get("p_value", 1)
    specs = main.get("specifications", [])

    out = []

    def S(title):
        out.append({"title": title, "blocks": []})

    def B(kind, text="", **kw):
        kw["kind"] = kind
        if text:
            kw["text"] = text
        out[-1]["blocks"].append(kw)

    # ═══════════════════════════════════════════════════════════════
    # 摘要 (Abstract)
    # ═══════════════════════════════════════════════════════════════
    S("摘要")
    custom_abstract = sec.get("abstract", "")
    if custom_abstract:
        B("paragraph", custom_abstract)
    elif coef:
        direction = "提高" if coef > 0 else "降低"
        abs_lines = []
        abs_lines.append(f"本文评估了{policy}对{outcome}的因果效应。")
        theory = chain.get("theoretical", "")
        final = chain.get("final", "")
        if theory:
            abs_lines.append(f"基于政策分配机制，本文采用{final or theory}作为核心识别策略。")
        abs_lines.append(
            f"基准回归结果表明，{policy}显著{direction}了{outcome}（"
            f"β = {coef:.4f}, SE = {se_val:.4f}, {'p < 0.001' if pval < 0.001 else 'p = ' + f'{pval:.4f}'}"
            f"），效应约{(2.71828**coef - 1) * 100:+.1f}%。"
        )
        n_t = meta.get("n_treated", "?")
        n_c = meta.get("n_control", "?")
        span = meta.get("time_span", meta.get("span", ""))
        if span and n_t and n_c:
            abs_lines.append(f"本文使用{span}中国{n_t}个处理组和{n_c}个对照组的面板数据。")

        # Event study summary
        pt = es.get("pre_trends_test", {})
        if pt:
            fv = pt.get("f_stat", 0)
            pv = pt.get("p_value", 1)
            if pv > 0.05:
                abs_lines.append("事件研究结果支持平行趋势假设。")
            else:
                abs_lines.append(f"事件研究结果显示平行趋势假设未通过（F={fv:.3f}, p={_p(pv)}）。")

        # Robustness summary
        robustness = O("robustness", [])
        if robustness:
            n_pass = sum(1 for r in robustness if r.get("passed"))
            abs_lines.append(f"稳健性检验（{n_pass}/{len(robustness)}项通过）进一步支持了主要结论。")

        # Causal claim
        strength = O("causal_claim_strength", "")
        if strength:
            strength_cn = {"strong": "强", "moderate": "中等", "suggestive": "弱", "not identifiable": "不可识别"}
            abs_lines.append(f"因果推断可信度评级：{strength_cn.get(strength, strength)}。")

        B("paragraph", "\n\n".join(abs_lines))

    # ═══════════════════════════════════════════════════════════════
    # 一、引言
    # ═══════════════════════════════════════════════════════════════
    S("引言")

    # 1.1 研究背景
    B("heading", "研究背景", level=2)
    bg = sec.get("intro_background", "")
    if bg.strip():
        for p in bg.split("\n\n"):
            if p.strip():
                B("paragraph", p.strip())
    else:
        B("paragraph", f"{policy}是中国近年来的一项重要政策改革。本研究评估其对{outcome}的因果效应。")

    # 1.2 文献综述
    B("heading", "文献综述", level=2)
    lit = sec.get("intro_literature", "")
    if lit.strip():
        for p in lit.split("\n\n"):
            if p.strip():
                B("paragraph", p.strip())
    else:
        B("paragraph", f"已有文献对该政策效果进行了初步探讨。本文使用更全面的数据和更严格的因果识别方法提供新证据。")

    # ═══════════════════════════════════════════════════════════════
    # 二、制度背景与理论分析
    # ═══════════════════════════════════════════════════════════════
    S("制度背景与理论分析")

    # 2.1 制度背景
    B("heading", "制度背景", level=2)
    institution = sec.get("institution", "")
    if institution.strip():
        for p in institution.split("\n\n"):
            if p.strip():
                B("paragraph", p.strip())
    else:
        B("paragraph", f"{policy}采取分批推广或统一实施方式，为因果识别提供了变异来源。")

    # 2.2 理论机制
    B("heading", "理论机制", level=2)
    theory_text = sec.get("theory", "")
    if theory_text.strip():
        for p in theory_text.split("\n\n"):
            if p.strip():
                B("paragraph", p.strip())
    else:
        B("paragraph", f"{policy}可能通过多种渠道影响{outcome}，各渠道的净效应需要通过实证识别。")

    # ═══════════════════════════════════════════════════════════════
    # 三、研究设计
    # ═══════════════════════════════════════════════════════════════
    S("研究设计")

    # 3.1 数据来源与样本
    B("heading", "数据来源与样本", level=2)
    data_desc = sec.get("data_desc", "")
    if data_desc.strip():
        B("paragraph", data_desc.strip())
    else:
        source = meta.get("source", "")
        data_path = meta.get("data_path", "")
        n_t = meta.get("n_treated", 0)
        n_c = meta.get("n_control", 0)
        n_all = meta.get("n_cities", n_t + n_c)
        n_y = meta.get("n_years", "?")
        span = meta.get("time_span", f"{n_y}年")
        n_obs = meta.get("n_obs", "?")
        el = "省级行政区" if (isinstance(n_all, int) and n_all <= 35) else "地级及以上城市"

        desc_parts = [f"本文使用{span}中国{n_all}个{el}的面板数据"]
        if isinstance(n_t, int) and isinstance(n_c, int) and n_t > 0 and n_c > 0:
            desc_parts[0] += f"（其中{n_t}个处理组，{n_c}个对照组）"
        desc_parts[0] += f"，共{n_obs}个观测值。"

        # Add data source if meaningful
        if source and source not in ("cs", "did", "twfe", "psm", "scm", "iv", "rdd", ""):
            desc_parts.append(f"数据来源：{source}。")
        if data_path:
            desc_parts.append(f"分析数据文件：{Path(data_path).name}。")

        B("paragraph", " ".join(desc_parts))

    # 3.2 变量定义
    B("heading", "变量定义", level=2)
    vars_list = model.get("variables", [])
    if vars_list:
        is_categorized = any(len(v) >= 3 for v in vars_list)
        has_source = any(len(v) > 3 and v[3] and v[3].strip() for v in vars_list)

        if is_categorized:
            rows = []
            for v in vars_list:
                cat = v[0]
                name = v[1] if len(v) > 1 else ""
                var_desc = v[2] if len(v) > 2 else ""
                source = v[3] if len(v) > 3 else ""
                row = [cat, name, var_desc]
                if has_source:
                    row.append(source if source else "—")
                rows.append(row)
            headers = ["变量类别", "变量名称", "定义"] + (["数据来源"] if has_source else [])
            B("table", headers=headers, rows=rows, caption="变量定义表")
        else:
            rows = []
            for v in vars_list:
                name = v[0]
                desc = v[1] if len(v) > 1 else ""
                src = v[2] if len(v) > 2 else ""
                row = [name, desc]
                if has_source:
                    row.append(src if src else "—")
                rows.append(row)
            headers = ["变量名称", "定义"] + (["数据来源"] if has_source else [])
            B("table", headers=headers, rows=rows, caption="变量定义表")
    else:
        controls = main.get("controls", [])
        rows = [["被解释变量", outcome], ["核心解释变量", main.get("treatment_var", "Treated × Post")]]
        for c in controls:
            rows.append([c, "控制变量"])
        B("table", headers=["变量名称", "定义"], rows=rows)

    # 3.3 描述性统计
    B("heading", "描述性统计", level=2)
    desc_vars = desc.get("variables", [])
    if desc_vars:
        n_t = desc.get("n_treated", "?")
        n_c = desc.get("n_control", "?")
        B("paragraph", f"样本包含{n_t}个处理组单元和{n_c}个对照组单元，共{meta.get('n_obs', '?')}个观测值。")
        rows = []
        for v in desc_vars:
            t = v.get("treated", 0) or 0
            c = v.get("control", 0) or 0
            rows.append([v.get("label", ""), f"{t:.3f}", f"{c:.3f}", f"{t - c:+.3f}",
                         f"{v.get('std_diff', 0):.3f}"])
        B("table", headers=["变量", "处理组均值", "对照组均值", "差异", "|标准化差异|"],
          rows=rows, caption="描述性统计与平衡性检验",
          note=desc.get("note", ""))
    else:
        B("paragraph", "描述性统计需要提供面板数据文件（使用 --data 参数）。")

    # 3.4 识别策略与模型设定
    B("heading", "识别策略与模型设定", level=2)
    eq = model.get("equation", r"Y = \alpha + \beta \cdot Treated \times Post + \gamma X + \mu_i + \lambda_t + \varepsilon")
    B("equation", eq)
    desc_t = model.get("description", "")
    if desc_t:
        B("paragraph", desc_t)
    ident = model.get("identification", "")
    if ident:
        B("paragraph", ident)

    # ═══════════════════════════════════════════════════════════════
    # 四、实证结果
    # ═══════════════════════════════════════════════════════════════
    S("实证结果")

    # 4.1 基准回归结果
    B("heading", "基准回归结果", level=2)
    if coef:
        direction = "提高" if coef > 0 else "降低"
        effect_pct = (2.71828 ** coef - 1) * 100
        B("paragraph",
          f"表1报告了基准回归结果。{policy}显著{direction}了{outcome}"
          f"（β = {coef:.4f}, SE = {se_val:.4f}，在{_sig(pval)}水平上显著），"
          f"效应约{effect_pct:+.1f}%（对数点解释）。")
    if specs:
        n = len(specs)
        headers = ["变量"] + [s.get("label", f"({i + 1})") for i, s in enumerate(specs)]
        rows = []
        rows.append(["Treated × Post"] + [_coef(s["coefficient"], s.get("p_value")) for s in specs])
        rows.append([""] + [_se(s["std_error"]) for s in specs])
        all_ctrls = []
        seen = set()
        for s in specs:
            for c in s.get("controls", []):
                if c not in seen:
                    all_ctrls.append(c)
                    seen.add(c)
        for ctrl in all_ctrls:
            r1 = [ctrl]
            r2 = [""]
            have_any = False
            for s in specs:
                cc = s.get("control_coefs", {}).get(ctrl)
                if cc:
                    r1.append(_coef(cc["coefficient"], cc.get("p_value")))
                    r2.append(_se(cc["std_error"]))
                    have_any = True
                else:
                    r1.append("")
                    r2.append("")
            if have_any:
                rows.append(r1)
                rows.append(r2)
        rows.append(["固定效应"] + ["是"] * n)
        rows.append(["观测值"] + [str(s.get("n_obs", "")) for s in specs])
        r2_val = main.get("r2", specs[-1].get("r2"))
        if r2_val is not None:
            rows.append(["R²"] + [f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val)] + [""] * (n - 1))
        B("table", headers=headers, rows=rows,
          caption=f"{policy}对{outcome}的回归结果",
          note="括号内为聚类稳健标准误。* p<0.1, ** p<0.05, *** p<0.01。GDP、人口等绝对值较大的控制变量建议使用对数值以获得弹性解释。")

    # 4.2 事件研究
    B("heading", "事件研究（平行趋势检验）", level=2)
    if es and es.get("coefficients"):
        pt = es.get("pre_trends_test", {})
        fv = pt.get("f_stat", 0)
        pv = pt.get("p_value", 1)
        passed = "支持" if pv > 0.05 else "不支持"
        B("paragraph",
          f"处理前各期系数联合F检验结果{passed}平行趋势假设"
          f"（F = {fv:.3f}, p = {_p(pv)}）。")
        rows = []
        for t in sorted(es["coefficients"].keys(), key=_safe_sort_key):
            c = es["coefficients"][t]
            pv_c = c.get("p_value")
            if pv_c is None:
                pv_c = _normal_pval(c.get("coefficient", 0), c.get("std_error", 0))
            rows.append([f"t={t}", _coef(c.get("coefficient", 0), pv_c),
                         _se(c.get("std_error", 0)), _p(pv_c)])
        B("table", headers=["相对时间", "系数", "标准误", "p值"], rows=rows,
          caption="事件研究估计结果",
          note=f"参考期：t=−1。N = {es.get('n_obs', '—')}。")
    elif es and es.get("att_by_period"):
        # Causal forest / DML style period-specific ATTs
        periods = es.get("att_by_period", [])
        if periods:
            rows = []
            for pe in periods:
                p_val = pe.get("p_value")
                if p_val is None:
                    p_val = _normal_pval(pe.get("att", 0), pe.get("se", 0))
                rows.append([str(pe.get("period", "")),
                             _coef(pe.get("att", 0), p_val),
                             _se(pe.get("se", 0)),
                             _p(p_val)])
            B("table", headers=["时期", "ATT", "标准误", "p值"], rows=rows,
              caption="各期处理效应")
    else:
        B("paragraph", "事件研究结果暂未提供。请确认事件研究脚本已成功运行。")

    # 4.3 稳健性检验
    S("稳健性检验")
    robustness = O("robustness", [])
    if robustness:
        n_pass = sum(1 for r in robustness if r.get("passed"))
        n_total = len(robustness)
        B("paragraph",
          f"本文进行了{n_total}项稳健性检验，其中{n_pass}项通过，{n_total - n_pass}项未通过或不明确。")

        # Group by type
        checks_list = []
        for r in robustness:
            name = r.get("name", "Unknown check")
            passed = r.get("passed", False)
            interp = r.get("interpretation", "")
            icon = "Pass" if passed else "Fail"
            checks_list.append([icon, name, "通过" if passed else "未通过", interp])
        B("table", headers=["", "检验", "结果", "说明"], rows=checks_list,
          caption="稳健性检验结果",
          note="稳健性检验用于评估基准回归结果对模型设定、样本选择和变量测量的敏感性。")

        # Oster bounds special handling
        sensitivity = O("sensitivity", {})
        if sensitivity or O("oster_delta"):
            oster_delta = O("oster_delta") or (sensitivity or {}).get("oster", {}).get("delta")
            if oster_delta is not None:
                B("paragraph",
                  f"Oster（2019）检验表明，未观测因素与已观测因素的选择比需达到δ = {oster_delta:.2f} "
                  f"才能完全解释处理效应。{'δ > 1，结果较为稳健。' if float(oster_delta) > 1 else 'δ ≤ 1，结果对遗漏变量较为敏感。'}")
    else:
        B("paragraph", "稳健性检验结果暂未提供。请使用 --stage8 和 --stage8-placebo 参数传入检验结果。")

    # 4.4 方法选择链（仅在方法变更时显示）
    fallback_attempts = O("fallback_attempts", [])
    if fallback_attempts and len(fallback_attempts) > 0:
        S("方法选择链")
        B("paragraph", "以下列出了方法确认过程中尝试的识别策略及其结果：")
        for i, fb in enumerate(fallback_attempts, 1):
            method_name = fb.get("method", f"方法{i}")
            outcome_status = fb.get("outcome", fb.get("status", ""))
            reason = fb.get("reason", "")
            B("paragraph",
              f"**{i}. {method_name}** — {outcome_status}"
              + (f"：{reason}" if reason else ""))

    # 4.5 因果推断可信度
    S("因果推断可信度评估")
    strength = O("causal_claim_strength", "not assessed")
    strength_cn = {"strong": "强（Strong）", "moderate": "中等（Moderate）",
                   "suggestive": "弱（Suggestive）", "not identifiable": "不可识别（Not Identifiable）"}
    B("paragraph", f"综合所有检验结果，本研究的因果推断可信度评级为：**{strength_cn.get(strength, strength)}**。")

    # Explain the rating
    assumptions = O("assumptions", [])
    n_pass = sum(1 for a in assumptions if a.get("verdict") == "PASS")
    n_fail = sum(1 for a in assumptions if a.get("verdict") == "FAIL")
    n_uncertain = sum(1 for a in assumptions if a.get("verdict") == "UNCERTAIN")
    B("paragraph",
      f"识别假设检验：{n_pass}项通过，{n_fail}项未通过，{n_uncertain}项无法检验。"
      f"方法变更：{'是' if chain.get('changed') else '否'}。")

    # ═══════════════════════════════════════════════════════════════
    # 五、结论与政策建议
    # ═══════════════════════════════════════════════════════════════
    conclusion = sec.get("conclusion", "")
    if conclusion.strip():
        S("结论与政策建议")
        for p in conclusion.split("\n\n"):
            if p.strip():
                B("paragraph", p.strip())
    elif coef:
        S("结论与政策建议")
        direction = "提高" if coef > 0 else "降低"
        auto_conclusion = (
            f"本文利用{policy}的准自然实验变异，使用{chain.get('final', chain.get('theoretical', '因果推断方法'))}"
            f"估计了{policy}对{outcome}的因果效应。"
            f"基准回归结果表明，{policy}显著{direction}了{outcome}"
            f"（β = {coef:.4f}, {'p < 0.001' if pval < 0.001 else f'p = {pval:.4f}'}）。"
        )
        # Add event study findings
        pt = es.get("pre_trends_test", {})
        if pt:
            pv = pt.get("p_value", 1)
            if pv > 0.05:
                auto_conclusion += "事件研究结果支持平行趋势假设，增强了因果推断的可信度。"
            else:
                auto_conclusion += "平行趋势假设未通过，结果需谨慎解读。"

        # Add policy implications
        auto_conclusion += (
            f"\n\n本研究的政策含义是清晰的：{policy}对{outcome}产生了显著的"
            f"{'正向' if coef > 0 else '负向'}影响。"
            f"政策制定者在评估{policy}的综合效果时，应将这一因果效应纳入考量。"
        )

        # Add limitations
        limitations = O("limitations", [])
        if limitations:
            auto_conclusion += "\n\n本研究存在以下局限："
            for i, lim in enumerate(limitations, 1):
                lim_text = lim if isinstance(lim, str) else lim.get("description", str(lim))
                if lim_text.strip():
                    auto_conclusion += f"（{i}）{lim_text.strip()}；"
            auto_conclusion = auto_conclusion.rstrip("；") + "。"
            auto_conclusion += "未来研究可以进一步处理这些局限，提供更精确的因果效应估计。"

        B("paragraph", auto_conclusion)

    # ═══════════════════════════════════════════════════════════════
    # 参考文献
    # ═══════════════════════════════════════════════════════════════
    S("参考文献")
    refs = O("references", [])
    ref_lines = []
    for i, r in enumerate(refs, 1):
        if isinstance(r, dict):
            ref_lines.append(f"[{i}] {r.get('text', '')}")
        else:
            ref_lines.append(f"[{i}] {r}")
    if ref_lines:
        B("paragraph", "\n\n".join(ref_lines))
    else:
        B("paragraph", "（参考文献待补充）")

    return out


# ═══════════════════════════ Markdown Renderer ═══════════════════════════
def render_markdown(data):
    sections = _build_report(data)
    title = f"{data.get('policy', 'Policy')}对{data.get('outcome', 'Outcome')}的影响：基于因果推断的实证评估"
    out = [f"# {title}", "",
           f"*生成日期：{data.get('generated', datetime.now().strftime('%Y-%m-%d'))}*",
           "", "---", ""]

    tn = [0]

    def _tn():
        tn[0] += 1
        return tn[0]

    def _md(h, r):
        L = ["| " + " | ".join(h) + " |", "| " + " | ".join(":---:" for _ in h) + " |"]
        for rw in r:
            cells = []
            for c in rw:
                s = str(c).replace("|", "\\|")
                # Insert <br> every ~35 chars for cells longer than 50 chars
                if len(s) > 50:
                    parts = []
                    remaining = s
                    while len(remaining) > 35:
                        # Find a natural break point (punctuation or space)
                        split_at = 35
                        for ch in "。，；、；）":
                            idx = remaining[:40].rfind(ch)
                            if idx > 20:
                                split_at = idx + 1
                                break
                        parts.append(remaining[:split_at])
                        remaining = remaining[split_at:]
                    if remaining:
                        parts.append(remaining)
                    s = "<br>".join(parts)
                cells.append(s)
            L.append("| " + " | ".join(cells) + " |")
        return "\n".join(L)

    sn = 0
    for sec in sections:
        sn += 1
        label = _SN[sn - 1] if sn <= 10 else str(sn)
        out.append(f"## {label}、{sec['title']}")
        out.append("")

        for b in sec["blocks"]:
            k = b["kind"]
            if k == "heading":
                depth = b.get("level", 2) + 2
                # Sub-headings get 中文数字.阿拉伯数字 format
                if b.get("level", 2) == 2 and sn <= 10:
                    sub_label = f"{label}.{_SN[len([x for x in sec['blocks'][:sec['blocks'].index(b)] if x['kind'] == 'heading'])]}"
                    out.append(f"{'#' * depth} {sub_label} {b['text']}")
                else:
                    out.append(f"{'#' * depth} {b['text']}")
                out.append("")
            elif k == "paragraph":
                prefix = "**" if b.get("bold") else ""
                suffix = "**" if b.get("bold") else ""
                out.append(f"{prefix}{b['text']}{suffix}")
                out.append("")
            elif k == "spacer":
                out.append("")
            elif k == "equation":
                out.append(f"```\n{b['text']}\n```")
                out.append("")
            elif k == "table":
                cap = b.get("caption", "")
                if cap:
                    out.append(f"**表{_tn()}：{cap}**")
                    out.append("")
                out.append(_md(b["headers"], b["rows"]))
                out.append("")
                if b.get("note"):
                    out.append(f"*注：{b['note']}*")
                    out.append("")

    return "\n".join(out)


# ═══════════════════════════ LaTeX Renderer ═══════════════════════════
def _te(t):
    """Escape text for LaTeX — preserves math commands."""
    s = str(t)
    for ch, esc in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                    ("_", "\\_"), ("~", "\\textasciitilde "), ("^", "\\textasciicircum ")]:
        s = s.replace(ch, esc)
    return s


def _is_numeric_cell(cell_str):
    """Check if a cell contains a numeric value (possibly with significance stars)."""
    s = str(cell_str).strip()
    if not s:
        return False
    # Strip significance stars
    s_clean = s.rstrip("*")
    # Check if it's a pure number
    if re.match(r'^[+-]?\d+\.?\d*$', s_clean):
        return True
    # Check scientific notation
    if re.match(r'^[+-]?\d+\.?\d*[eE][+-]?\d+$', s_clean):
        return True
    return False


def render_latex(data):
    sections = _build_report(data)
    title = f"{_te(data.get('policy', 'Policy'))}对{_te(data.get('outcome', 'Outcome'))}的影响：基于因果推断的实证评估"

    L = ["% !TEX program = xelatex",
         "\\documentclass[11pt,a4paper]{article}",
         "\\usepackage{booktabs,siunitx,geometry,hyperref,caption,amsmath}",
         "\\usepackage{fontspec,xeCJK}",
         "\\setCJKmainfont{SimSun}[BoldFont=SimHei]",
         "\\setmainfont{Times New Roman}",
         "\\geometry{margin=1in}",
         "\\captionsetup{font=small,labelfont=bf}",
         f"\\title{{{title}}}",
         "\\author{}",
         f"\\date{{{data.get('generated', '')}}}",
         "\\begin{document}",
         "\\maketitle",
         ""]

    for sec in sections:
        L.append(f"\\section{{{_te(sec['title'])}}}")

        for b in sec["blocks"]:
            k = b["kind"]
            if k == "heading":
                lvl = b.get("level", 2)
                cmd = "\\subsection" if lvl == 2 else "\\subsubsection"
                L.append(f"{cmd}{{{_te(b['text'])}}}")
                L.append("")
            elif k == "paragraph":
                t = _te(b["text"])
                if b.get("bold"):
                    t = f"\\textbf{{{t}}}"
                L.append(t)
                L.append("")  # Blank line for paragraph separation
            elif k == "spacer":
                L.append("")
            elif k == "equation":
                eq = b["text"]
                if "\\" in eq or "{" in eq:
                    L.append(f"\\[{eq}\\]")
                else:
                    L.append(f"\\[{_te(eq)}\\]")
                L.append("")
            elif k == "table":
                cap = b.get("caption", "")
                rows = b["rows"]
                headers = b["headers"]
                n_cols = len(headers)
                if cap:
                    L.append("\\begin{table}[ht]")
                    L.append("\\centering")
                    L.append(f"\\caption{{{_te(cap)}}}")

                # Column widths proportional to content length
                col_lengths = []
                for ci in range(n_cols):
                    max_len = max(
                        (len(str(r[ci])) if ci < len(r) else 0)
                        for r in rows
                    )
                    max_len = max(max_len, len(str(headers[ci])))
                    col_lengths.append(max_len)
                total_len = sum(col_lengths) or 1
                col_is_num = [all(_is_numeric_cell(r[ci]) if ci < len(r) else False
                                  for r in rows) for ci in range(n_cols)]

                fmt_parts = ["@{}"]
                for ci in range(n_cols):
                    if col_is_num[ci] and n_cols <= 6:
                        fmt_parts.append("S[table-format=-1.6]")
                    else:
                        w = 0.04 + (col_lengths[ci] / total_len) * 0.84
                        fmt_parts.append(f"p{{{w:.3f}\\linewidth}}")
                fmt_parts.append("@{}")
                fmt = "".join(fmt_parts)

                L.append(f"\\begin{{tabular}}{{{fmt}}}")
                L.append("\\toprule")
                L.append(" & ".join(f"{{{_te(h)}}}" for h in headers) + " \\\\")
                L.append("\\midrule")

                for r in rows:
                    cells = []
                    for ci, c in enumerate(r):
                        c_str = str(c)
                        if not c_str.strip():
                            cells.append("")
                        elif ci > 0 and _is_numeric_cell(c_str):
                            cells.append(c_str)
                        else:
                            cells.append(_te(c_str))
                    L.append(" & ".join(cells) + " \\\\")

                L.append("\\bottomrule")
                if b.get("note"):
                    L.append(
                        f"\\multicolumn{{{n_cols}}}{{@{{}}p{{0.9\\linewidth}}}}{{\\footnotesize {_te(b['note'])}}} \\\\")
                L.append("\\end{tabular}")
                if cap:
                    L.append("\\end{table}")
                L.append("")

    L.append("\\end{document}")
    return "\n".join(L)


# ═══════════════════════════ Compile ═══════════════════════════
def _compile_latex(tex_path):
    import subprocess, shutil
    xelatex = shutil.which("xelatex")
    if not xelatex:
        print("  [WARN] xelatex not found, skipping PDF compilation")
        return False
    try:
        tex_file = Path(tex_path).name
        tex_dir = str(Path(tex_path).parent)
        for _ in range(2):
            subprocess.run(
                [xelatex, "-interaction=nonstopmode", tex_file],
                capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace", cwd=tex_dir,
            )
        # Clean aux files
        for ext in [".aux", ".log", ".out"]:
            aux = Path(tex_path).with_suffix(ext)
            if aux.exists():
                try:
                    aux.unlink()
                except Exception:
                    pass
        return Path(tex_path).with_suffix(".pdf").exists()
    except Exception as e:
        print(f"  [WARN] Compilation error: {e}")
        return False


# ═══════════════════════════ CLI ═══════════════════════════
def main():
    p = argparse.ArgumentParser(description="学术报告渲染器 (Markdown + XeLaTeX)")
    p.add_argument("--data", required=True, help="Path to report_data.json (from output_report.py)")
    p.add_argument("--md", default=None, help="Output path for Markdown")
    p.add_argument("--tex", default=None, help="Output path for XeLaTeX")
    p.add_argument("--compile", action="store_true", default=True,
                   help="Compile LaTeX to PDF (default: on if xelatex found)")
    p.add_argument("--no-compile", action="store_false", dest="compile",
                   help="Skip PDF compilation")
    args = p.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)

    policy = data.get("policy", "report")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path.home() / "Desktop" / "policy_eval_output"
    subject_dir = _find_or_create_subject_dir(policy, output_base)
    base = subject_dir / ts
    md_dir = base / "markdown"
    tex_dir = base / "latex"
    md_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    md_path = Path(args.md) if args.md else (md_dir / "paper.md")
    md = render_markdown(data)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown → {md_path}")

    tex_path = Path(args.tex) if args.tex else (tex_dir / "paper.tex")
    tex = render_latex(data)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")
    print(f"XeLaTeX → {tex_path}")

    if args.compile:
        print("Compiling PDF...")
        if _compile_latex(str(tex_path)):
            print(f"PDF → {tex_path.with_suffix('.pdf')}")
        else:
            print("PDF compilation failed (xelatex may not be installed)")


if __name__ == "__main__":
    main()
