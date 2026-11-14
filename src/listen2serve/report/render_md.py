"""Render multi-turn evaluation results as Markdown and JSON."""

from __future__ import annotations

from typing import Any


def render_md_report(aggregated: dict[str, Any], title: str = "Listen2Serve 评测报告") -> str:
    """渲染分层聚合 Markdown 报告。"""
    summary = aggregated.get("summary", {})
    lines = [f"# {title}", ""]
    lines += _render_provenance(aggregated.get("provenance"))
    lines.append(f"总样本数：{summary.get('n', 0)}")
    lines.append(
        f"完整通话：{summary.get('n_complete', summary.get('n', 0))}；"
        f"端点截断、关键轮可评：{summary.get('n_partial', 0)}。"
    )
    lines.append(
        "KeyTurnPass / VoicePass 包含已完整取得关键轮的截断通话；"
        "FlowRate / TaskScore 仅汇总完整通话，截断通话的全程指标为 N/A。"
    )
    lines.append(
        f"有效分母：KeyTurnPass={summary.get('key_turn_n', 0)}，"
        f"VoicePass={summary.get('voice_n', 0)}，"
        f"FlowRate={summary.get('flow_n', 0)}，TaskScore={summary.get('task_n', 0)}。"
    )
    lines.append("")
    lines.append("## 总体指标")
    lines.append("")
    lines.append("**主指标**")
    lines.append("")
    lines.append("| 指标 | 回答的问题 | 值 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| KeyTurnPass（关键轮应对通过率，0-1） | 关键轮策略是否回应用户状态 | {_pct_or_na(summary.get('key_turn_pass'))} |"
    )
    lines.append(
        f"| VoicePass（声音通过率，0-1） | 三维均分 ≥3.5 且各维 ≥3 | {_pct_or_na(summary.get('voice_pass'))} |"
    )
    lines.append(
        f"| KeyTurnScore（弦外之音量规均分，MOS 1-5） | 对用户真实状态的识别与承接质量 | {_score_or_na(summary.get('key_turn_score_mean'))} |"
    )
    lines.append(
        f"| FlowRate（过程清单命中比例，部分计 0.5，0-1） | 全对话必需动作的完成程度 | {_score_or_na(summary.get('flow_rate_mean'))} |"
    )
    lines.append(
        f"| TaskScore（终局结果，0-2 归一 0/0.5/1） | 对话任务的达成程度 | {_score_or_na(summary.get('task_score_mean'))} |"
    )
    lines.append(
        f"| VoiceMean（三维均分，MOS 1-5） | 角色匹配、状态适配和自然度的平均分 | {_score_or_na(summary.get('voice_mean'))} |"
    )
    lines.append(
        f"| Voice 角色声音匹配 / 状态声音适配 / 自然度 | — | "
        f"{_score_or_na(summary.get('voice_role_match_mean'))} / "
        f"{_score_or_na(summary.get('voice_state_fit_mean'))} / "
        f"{_score_or_na(summary.get('voice_naturalness_mean'))} |"
    )
    lines.append("")
    lines.append("**质量监控量（不算被测成绩）**")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(
        f"| VocabViolationRate（用户话术词表，全轮） | {_pct_or_na(summary.get('vocab_violation_rate'))} |"
    )
    lines.append("")
    lines.append("## 分层结果（角色 × 提示条件 × 动态）")
    lines.append("")
    lines.append(
        "| 角色 | 提示条件 | 措辞 | 动态 | N | 完整/截断 | Flow/Task N | KeyTurnPass | VoicePass | KeyTurnScore | FlowRate | TaskScore | VoiceMean | VocabViolation |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in aggregated.get("rows", []):
        lines.append(
            f"| {r['role']} | {r.get('condition') or '—'} | {r['wording']} | {r['dynamics']} | {r['n']} | "
            f"{r.get('n_complete', r['n'])}/{r.get('n_partial', 0)} | "
            f"{r.get('flow_n', 0)}/{r.get('task_n', 0)} | "
            f"{_pct_or_na(r.get('key_turn_pass'))} | {_pct_or_na(r.get('voice_pass'))} | {_score_or_na(r.get('key_turn_score_mean'))} | {_score_or_na(r.get('flow_rate_mean'))} | "
            f"{_score_or_na(r.get('task_score_mean'))} | {_score_or_na(r.get('voice_mean'))} | "
            f"{_pct_or_na(r.get('vocab_violation_rate'))} |"
        )
    return "\n".join(lines)


def _render_provenance(provenance: dict[str, Any] | None) -> list[str]:
    """Render the recorded code, data and model provenance."""
    if not provenance:
        return []
    lines = ["## 运行溯源", ""]
    if provenance.get("manifest") != "ok":
        lines.append(
            f"> ⚠ manifest 缺失：{provenance.get('run_id', '?')} 无 run_manifest.json，代码/数据版本不可追溯"
        )
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
        f"| 请求的被测模型 | {models.get('target') or '—'} |",
        f"| 请求的演绎模型 | {models.get('sim') or '—'} |",
        f"| 请求的裁判模型 | {models.get('judge') or '—'}（voice: {models.get('voice_judge') or '—'}） |",
        f"| 服务端报告的裁判模型 | {models.get('judge_response_model') or '未报告或多个型号，详见 JSON'}（voice: {models.get('voice_judge_response_model') or '未报告或多个型号，详见 JSON'}） |",
        f"| 请求的 TTS | {models.get('tts') or '—'} |",
        f"| 生成时间 | {provenance.get('created_at') or '—'} |",
        "",
    ]
    if models.get("judge_generation_claim"):
        lines.append(
            f"> ⚠ 当前报告请求的裁判为 `{models.get('judge') or '—'}`；"
            f"生成步曾登记 `{models['judge_generation_claim']}`"
            f"（该字段写于评测之前，不代表实际裁判）。"
        )
        lines.append("")
    return lines


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _pct_or_na(v: float | None) -> str:
    """Format an observed rate or show N/A."""
    return _pct(v) if v is not None else "N/A"


def _score_or_na(v: float | None) -> str:
    """Format an observed score or show N/A."""
    return f"{v:.2f}" if v is not None else "N/A"
