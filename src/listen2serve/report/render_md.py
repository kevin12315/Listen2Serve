"""报告渲染：Markdown（论文 Table 1-4 对齐）与 JSON 导出。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_md_report(aggregated: dict[str, Any], title: str = "Listen2Serve 评测报告") -> str:
    """渲染分层聚合 Markdown 报告。"""
    summary = aggregated.get("summary", {})
    lines = [f"# {title}", ""]
    lines += _render_provenance(aggregated.get("provenance"))
    lines.append(f"总样本数：{summary.get('n', 0)}")
    lines.append("")
    lines.append("## 总体指标")
    lines.append("")
    lines.append("**主指标（v2.3-J：量纲统一，文本链 0-1 / 声音主观 MOS 1-5）**")
    lines.append("")
    lines.append("| 指标 | 回答的问题 | 值 |")
    lines.append("|---|---|---|")
    lines.append(f"| KeyTurnPass（关键轮应对通过率，0-1） | 压力点上接住了没 | {_pct_or_na(summary.get('key_turn_pass'))} |")
    lines.append(f"| KeyTurnScore（弦外之音量规均分，MOS 1-5） | 语气里的真意听出来并接住了没 | {_score_or_na(summary.get('key_turn_score_mean'))} |")
    lines.append(f"| KeyBehaviorMet（关键行为命中率，兼容派生） | — | {_pct_or_na(summary.get('key_behavior_met_rate'))} |")
    lines.append(f"| FlowRate（过程清单命中比例，部分计 0.5，0-1） | 规定动作做全了没 | {_score_or_na(summary.get('flow_rate_mean'))} |")
    lines.append(f"| TaskScore（终局结果，0-2 归一 0/0.5/1） | 这通电话办成了没 | {_score_or_na(summary.get('task_score_mean'))} |")
    lines.append(f"| VoiceMean（三维均分，MOS 1-5） | 听着像合格客服吗 | {_score_or_na(summary.get('voice_mean'))} |")
    lines.append(f"| Voice 角色声音匹配 / 状态声音适配 / 自然度 | — | "
                 f"{_score_or_na(summary.get('voice_role_match_mean'))} / "
                 f"{_score_or_na(summary.get('voice_state_fit_mean'))} / "
                 f"{_score_or_na(summary.get('voice_naturalness_mean'))} |")
    lines.append("")
    lines.append("**质量监控量（不算被测成绩）**")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| VocabViolationRate（用户话术词表，全轮） | {_pct_or_na(summary.get('vocab_violation_rate'))} |")
    # 退役名（v2.2-I 旧口径；仅旧 verdict 有值时渲染，决议）
    if summary.get("required_rate_mean") is not None or summary.get("policy_pass"):
        lines.append("")
        lines.append("**退役指标（v2.2-I 旧口径，仅供历史对照）**")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        lines.append(f"| RequiredRate 均值（旧） | {_pct_or_na(summary.get('required_rate_mean'))} |")
        lines.append(f"| PolicyPass（旧） | {_pct(summary.get('policy_pass', 0))} |")
        lines.append(f"| VoicePass | {_pct(summary.get('voice_pass', 0))} |")
        lines.append(f"| JointPass | {_pct(summary.get('joint_pass', 0))} |")
        lines.append(f"| TaskCompletion (0-2，旧) | {summary.get('task_completion_mean', 0)} |")
        lines.append(f"| TransitionScore (0-2，已退役) | {summary.get('transition_score_mean', 0)} |")
    lines.append("")
    lines.append("## 分层结果（角色 × 透明度 × 动态）")
    lines.append("")
    lines.append("| 角色 | 透明度 | 动态 | N | KeyTurnPass | KeyTurnScore | FlowRate | TaskScore | VoiceMean | VocabViolation |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in aggregated.get("rows", []):
        lines.append(
            f"| {r['role']} | {r['leakage_level']} | {r['dynamics']} | {r['n']} | "
            f"{_pct_or_na(r.get('key_turn_pass'))} | {_score_or_na(r.get('key_turn_score_mean'))} | {_score_or_na(r.get('flow_rate_mean'))} | "
            f"{_score_or_na(r.get('task_score_mean'))} | {_score_or_na(r.get('voice_mean'))} | "
            f"{_pct_or_na(r.get('vocab_violation_rate'))} |"
        )
    return "\n".join(lines)


def _render_provenance(provenance: dict[str, Any] | None) -> list[str]:
    """内部方案：报告头运行溯源（git_commit 前 8 位 / dataset_version / models）；
    旧 run 无 manifest 时显示「manifest 缺失（旧 run）」；字段缺失一律渲染 —。"""
    if not provenance:
        return []
    lines = ["## 运行溯源", ""]
    if provenance.get("manifest") != "ok":
        lines.append(f"> ⚠ manifest 缺失（旧 run）：{provenance.get('run_id', '?')} 无 run_manifest.json，代码/数据版本不可追溯")
        lines.append("")
        return lines
    models = provenance.get("models") or {}
    commit = provenance.get("git_commit") or "—"
    if provenance.get("git_dirty"):
        commit += "（含未提交改动）"
    elif provenance.get("fallback_tree_hash"):
        commit = f"无 git，fallback_tree_hash={provenance['fallback_tree_hash']}"
    lines += [
        "| 溯源项 | 值 |",
        "|---|---|",
        f"| run_id | {provenance.get('run_id', '—')} |",
        f"| 代码版本 | {commit} |",
        f"| 数据版本 | {provenance.get('dataset_version') or '—'} |",
        f"| 被测模型 | {models.get('target') or '—'} |",
        f"| 演绎模型 | {models.get('sim') or '—'} |",
        f"| 裁判模型 | {models.get('judge') or '—'}（voice: {models.get('voice_judge') or '—'}） |",
        f"| TTS | {models.get('tts') or '—'} |",
        f"| 生成时间 | {provenance.get('created_at') or '—'} |",
        "",
    ]
    if models.get("judge_generation_claim"):
        # 不静默取舍：生成期占位与实测裁判不一致时，把两个值都摊开写明来源，
        # 否则读者无法判断这一栏是"回写后的真值"还是"生成时的默认值"。
        lines.append(
            f"> ⚠ 裁判栏取**实测/回写值** `{models.get('judge') or '—'}`；"
            f"生成步曾登记 `{models['judge_generation_claim']}`"
            f"（该字段写于评测之前，不代表实际裁判）。"
        )
        lines.append("")
    return lines


def export_json(aggregated: dict[str, Any], out_path: str | Path) -> None:
    """导出聚合结果 JSON。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _pct_or_na(v: float | None) -> str:
    """v5.6：词表违规率无校验样本时为 None，渲染 N/A（避免旧批次假性 0%）。"""
    return _pct(v) if v is not None else "N/A"


def _score_or_na(v: float | None) -> str:
    """内部方案：连续主指标（Voice 1–5 分）无数据时渲染 N/A（旧 run/text 模态）。"""
    return f"{v:.2f}" if v is not None else "N/A"
