"""分层聚合：角色 × 透明度层级 × 状态动态 的指标聚合统计。

内部方案（v2.3-J）：主指标改名（决议）——对外主指标 = KeyTurnPass/FlowRate/
TaskScore/VoiceMean，量纲统一（文本 0-1 / 声音 MOS 1-5）；PolicyPass/RequiredRate/
task_completion/transition_score 名退役（旧键继续计算仅供旧 verdict 兼容渲染）。
新指标读 v2.3-J 增量键（key_turn_pass/flow_rate/task_score），旧 verdict 无新键
时返回 None（渲染 N/A）。

K18c：增操纵有效性指标（leakage_*）。与其它指标根本不同——它们不评客服，
而是度量“用户侧的 T1/T2/T3 是否真的演成了三个透明度层级”，是客服侧层间对比
可解读的前提：leakage_score_mean 若不呈 T1 > T2 > T3，则该批次的透明度对照不成立。
注意分层键用的是 leakage_level（explicit/implicit_consistent/prosody_only），
不是 leakage_label（T1/T2/T3）；两者一一对应且字母序同序。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# Voice 三维字段（voice_judge 输出，1–5 分）
VOICE_DIM_KEYS = ("role_voice_match", "state_voice_fit", "naturalness")


def aggregate_results(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """按 角色×透明度×动态 分层聚合。

    Args:
        verdicts: 每条场景的完整判定结果（含 scenario_id/role/leakage_level/dynamics
                  + policy/voice/joint + task_completion/transition_score）
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        key = (v.get("role", ""), v.get("leakage_level", ""), v.get("dynamics", ""))
        groups[key].append(v)

    rows = []
    for (role, leakage, dynamics), items in sorted(groups.items()):
        row = {
            "role": role,
            "leakage_level": leakage,
            "dynamics": dynamics,
            **metric_block(items),
        }
        rows.append(row)

    return {"summary": metric_block(verdicts), "rows": rows}


def metric_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    """一组 verdict 的全部指标（rows 单元格与 summary 共用同一口径）。

    抽成独立函数是为了让网页统计页的各条边际（按透明度／按角色／按动态）与
    report 的分层表严格同源——同一份 verdict 无论怎么切分，指标定义都不会漂移。
    """
    return {
        "n": len(items),
        # ---- 内部方案 主指标（v2.3-J；v2.5-K 增 KeyTurnScore）：文本链 0-1 + 量规 MOS 1-5，声音主观 MOS 1-5 ----
        "key_turn_pass": _rate_or_none(items, "key_turn_pass"),
        "key_turn_score_mean": _mean_or_none(items, "key_turn_score"),
        "key_behavior_met_rate": _hit_rate(items, "key_behavior_met"),
        "flow_rate_mean": _mean_or_none(items, "flow_rate"),
        "flow_rate_in_window_mean": _mean_or_none(items, "flow_rate_in_window"),
        "task_score_mean": _mean_or_none(items, "task_score"),
        "voice_mean": _voice_detail_mean(items, "mean"),
        "voice_role_match_mean": _voice_detail_mean(items, "role_voice_match"),
        "voice_state_fit_mean": _voice_detail_mean(items, "state_voice_fit"),
        "voice_naturalness_mean": _voice_detail_mean(items, "naturalness"),
        # ---- K18c 操纵有效性检验（与客服表现无关，只度量三层透明度自身是否分级）----
        "leakage_score_mean": _mean_or_none(items, "leakage_score"),
        "leakage_state_correct_rate": _rate_or_none(items, "leakage_state_correct"),
        "leakage_stance_marker_mean": _mean_or_none(items, "leakage_stance_markers"),
        # ---- 收尾轮确定性检查（closing_check；零 API 成本，可重算历史批次）----
        # 与其它指标的性质差别：它不评策略优劣，评的是「这通电话作为一通电话是否完整」。
        # 分母只含末轮用户出现词面收尾信号的场景（closing_dangling/closing_ok 在无信号时为
        # None，_rate_or_none 自动剔除），因此三项必须连读：signal 是分母覆盖率。
        "closing_signal_rate": _rate_or_none(items, "closing_signal"),
        "closing_dangling_rate": _rate_or_none(items, "closing_dangling"),
        "closing_ok_rate": _rate_or_none(items, "closing_ok"),
        # ---- 退役名（旧 verdict 兼容，仅供历史报告重渲染，新批次为 None）----
        # 内部方案d：原先这五个走会兜底成 0.0 的 _rate/_mean，使 v2.3-J 新批次的
        # 「未测量」在横向对照页显示成 0.0%；而旧批次真值本就贴近零（v2.2-I 实测
        # PolicyPass 0.0556），两者视觉上无法区分 —— 属「静默降级」同一类病。
        # 改走 _*_or_none：无值即 None，前端已有 ?? "—" 的 N/A 渲染。
        "required_rate_mean": _required_rate_mean(items),
        "policy_pass": _rate_or_none(items, "policy_pass"),
        "voice_pass": _rate_or_none(items, "voice_pass"),
        "joint_pass": _rate_or_none(items, "joint_pass"),
        "task_completion_mean": _mean_or_none(items, "task_completion"),
        "transition_score_mean": _mean_or_none(items, "transition_score"),
        "vocab_violation_rate": _vocab_violation_rate(items),
    }


def _level_of(v: dict[str, Any]) -> str:
    """透明度层名：优先 leakage_label（T1/T2/T3，论文口径且可读），回落 leakage_level。"""
    return str(v.get("leakage_label") or v.get("leakage_level") or "")


def aggregate_marginals(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """多组边际统计（网页统计页专用；口径与 aggregate_results 严格同源）。

    为什么要边际而不是只有 rows：rows 按 角色×透明度×动态 三元组切，18 条批次里
    每格只剩 1 条，读不出任何趋势。论文正文真正要看的是单维边际——尤其
    by_leakage 这一条，它同时承载「客服在 T1/T2/T3 上的表现差异」与
    「leakage_score 是否单调（该差异可否解读的前提）」两件事。

    返回 {overall, by_leakage, by_role, by_dynamics, by_role_leakage, by_model,
          cells, leakage_monotonic}；leakage_monotonic 在层数不足或探针缺值时为 None。
    by_model 只有在 verdict 带 model 字段时才有意义（单批次内恒为一个模型，
    跨批次合并的 verdicts 才会分出多行）。
    """
    by_leakage = _grouped(verdicts, _level_of)
    scores = [r["leakage_score_mean"] for r in by_leakage]
    monotonic = None
    if len(scores) > 1 and all(isinstance(s, (int, float)) for s in scores):
        monotonic = all(a > b for a, b in zip(scores, scores[1:]))
    return {
        "overall": metric_block(verdicts),
        "by_leakage": by_leakage,
        "by_role": _grouped(verdicts, lambda v: str(v.get("role") or "")),
        "by_dynamics": _grouped(verdicts, lambda v: str(v.get("dynamics") or "")),
        "by_role_leakage": _grouped(verdicts, lambda v: f"{v.get('role') or ''}／{_level_of(v)}"),
        "by_model": _grouped(verdicts, lambda v: str(v.get("model") or "")),
        "cells": aggregate_results(verdicts)["rows"],
        "leakage_monotonic": monotonic,
    }


def _grouped(verdicts: list[dict[str, Any]], keyfn) -> list[dict[str, Any]]:
    """按 keyfn 分组后逐组套 metric_block；组名升序（表格顺序稳定，便于逐版对账）。"""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in verdicts:
        buckets[keyfn(v)].append(v)
    return [{"group": k, **metric_block(items)} for k, items in sorted(buckets.items())]


# 配对检验取的连续主指标：都是 0-1 或 MOS，逐场景可直接相减。
# 布尔指标（key_turn_pass）不进差值表，改用胜/平/负计数体现。
PAIRED_KEYS = ("key_turn_score", "flow_rate", "task_score", "leakage_score")


def aggregate_cross_model(
    verdicts: list[dict[str, Any]], baseline: str = "",
) -> dict[str, Any]:
    """跨模型对照（n5）：多批次 verdicts 合并后按 model 切边际，并做同数据配对差值。

    为什么要配对而不是只比总均值：四个批次跑的是同一份 108 条场景，配对能把
    场景难度差异消掉——总均值相近也可能是「A 在 T1 强、B 在 T3 强」互相抵消，
    只有逐场景相减才看得出。未在两侧同时判定成功的场景一律排除（n_paired 会小于
    各自的 n），否则差值里混进的是「谁失败得多」而不是「谁答得好」。

    Args:
        verdicts: 合并后的判定结果，每条须带 model 与 scenario_id。
        baseline: 配对基准模型名；留空取字母序首个。
    Returns:
        {models, baseline, by_model, by_model_leakage, monotonic_by_model, paired}
    """
    models = sorted({str(v.get("model") or "") for v in verdicts} - {""})
    base = baseline if baseline in models else (models[0] if models else "")

    mono: dict[str, bool | None] = {}
    for m in models:
        sub = [v for v in verdicts if str(v.get("model") or "") == m]
        mono[m] = aggregate_marginals(sub)["leakage_monotonic"] if sub else None

    idx: dict[tuple[str, str], dict[str, Any]] = {}
    for v in verdicts:
        sid = str(v.get("scenario_id") or "")
        if sid:
            idx[(str(v.get("model") or ""), sid)] = v
    base_sids = [sid for (m, sid) in idx if m == base]

    paired = []
    for m in models:
        if m == base:
            continue
        pairs = [(idx[(base, sid)], idx[(m, sid)]) for sid in base_sids
                 if (m, sid) in idx]
        row: dict[str, Any] = {"model": m, "baseline": base, "n_paired": len(pairs)}
        for key in PAIRED_KEYS:
            diffs = [b[key] - a[key] for a, b in pairs
                     if isinstance(a.get(key), (int, float))
                     and isinstance(b.get(key), (int, float))]
            row[f"d_{key}"] = round(sum(diffs) / len(diffs), 4) if diffs else None
            row[f"n_{key}"] = len(diffs)
        wins = sum(1 for a, b in pairs if _kt(b) is not None and _kt(a) is not None
                   and _kt(b) > _kt(a))
        losses = sum(1 for a, b in pairs if _kt(b) is not None and _kt(a) is not None
                     and _kt(b) < _kt(a))
        row["win"], row["loss"] = wins, losses
        row["tie"] = len(pairs) - wins - losses
        paired.append(row)

    return {
        "models": models,
        "baseline": base,
        "by_model": _grouped(verdicts, lambda v: str(v.get("model") or "")),
        "by_model_leakage": _grouped(
            verdicts, lambda v: f"{v.get('model') or ''}／{_level_of(v)}"),
        "monotonic_by_model": mono,
        "paired": paired,
    }


def _kt(v: dict[str, Any]) -> float | None:
    """配对胜负判据取 KeyTurnScore（弦外之音量规，本研究的主结论指标）。"""
    x = v.get("key_turn_score")
    return x if isinstance(x, (int, float)) else None


def _rate_or_none(items: list[dict[str, Any]], key: str) -> float | None:
    """布尔指标通过率；无该键（旧 verdict）返回 None（渲染 N/A），不假性记 0。"""
    vals = [i.get(key) for i in items if i.get(key) is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals), 4)


def _mean_or_none(items: list[dict[str, Any]], key: str) -> float | None:
    """连续指标宏平均；无新键（旧 verdict）返回 None（渲染 N/A），不假性记 0。"""
    vals = [i.get(key) for i in items if isinstance(i.get(key), (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _hit_rate(items: list[dict[str, Any]], key: str) -> float | None:
    """key_behavior_met 命中率（是/部分 计命中）；旧 verdict 无该键返回 None。"""
    vals = [i.get(key) for i in items if i.get(key) is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v in ("是", "部分")) / len(vals), 4)


def _required_rate_mean(items: list[dict[str, Any]]) -> float | None:
    """连续主指标：逐场景 required_rate（policy_detail 明细聚合口径，hit=是 占比）
    的宏平均。旧 run 无 policy_detail.required_rate 时返回 None（渲染 N/A）。"""
    vals = [(i.get("policy_detail") or {}).get("required_rate") for i in items]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _voice_detail_mean(items: list[dict[str, Any]], key: str) -> float | None:
    """连续主指标：voice_detail 内字段（mean 或三维单项 1–5 分）的宏平均。
    无 voice_detail（如 text 模态 Voice 跳过）时返回 None（渲染 N/A）。"""
    vals = [(i.get("voice_detail") or {}).get(key) for i in items]
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _vocab_violation_rate(items: list[dict[str, Any]]) -> float | None:
    """非关键轮词表违规率（B3）：微平均（总违规轮/总校验轮）。

    v5.6：分母为 0（无校验轮，如旧批次 verdict 无该字段）时返回 None（报告渲染
    为 N/A），不再记 0，避免旧批次假性 0% 违规率。"""
    checked = sum(i.get("vocab_checked_turns") or 0 for i in items)
    violations = sum(i.get("vocab_violation_turns") or 0 for i in items)
    if not checked:
        return None
    return round(violations / checked, 4)
