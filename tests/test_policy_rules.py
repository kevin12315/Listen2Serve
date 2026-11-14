"""六态规则、资产状态标签归一化和轮次预算检查。"""

from listen2serve.domains import available_roles, get_domain
from listen2serve.domains.base import LEGACY_STATE_MAP, STATE_LABELS, normalize_state
from listen2serve.emotion import STATE_EMOTION


def test_policy_rules_cover_all_canonical_states():
    """三域 policy_rules 键集合均覆盖全部 6 个 canonical 状态。"""
    canonical = set(STATE_LABELS)
    assert len(canonical) == 6
    assert "neutral" in canonical
    for role in available_roles():
        domain = get_domain(role)
        assert set(domain.policy_rules) == canonical, (
            f"{role} policy_rules 覆盖缺口: 缺 {canonical - set(domain.policy_rules)}, "
            f"多 {set(domain.policy_rules) - canonical}"
        )


def test_policy_rules_three_part_template():
    """每条规则必须含 required/forbidden（三段式模板：必须/禁止/可选 + 声音要求）。"""
    for role in available_roles():
        domain = get_domain(role)
        for state, rules in domain.policy_rules.items():
            assert rules.get("required"), f"{role}/{state} 缺 required"
            assert rules.get("forbidden"), f"{role}/{state} 缺 forbidden"
            assert "allowed" in rules, f"{role}/{state} 缺 allowed"


def test_state_labels_align_with_vocab_emotion():
    """STATE_LABELS 与 emotion.STATE_EMOTION 使用同一套 canonical 英文码。"""
    assert set(STATE_LABELS) == set(STATE_EMOTION)


def test_normalize_state_new_format():
    """新格式 canonical 英文码原样返回。"""
    for code in STATE_LABELS:
        assert normalize_state(code) == code


def test_normalize_state_legacy_codes():
    """旧 11 态英文码经 LEGACY_STATE_MAP 归一化到新 6 态。"""
    for old, new in LEGACY_STATE_MAP.items():
        assert normalize_state(old) == new


def test_normalize_state_legacy_labels():
    """旧格式中文标签（含历史变体后缀）归一化到新 6 态码。"""
    assert normalize_state("N1 挫败/升级") == "displeased"
    assert normalize_state("N2 急迫/不便") == "urgent"
    assert normalize_state("N5 厌烦/抗拒") == "displeased"
    # 历史变体标签（marketing/hotline 旧键）
    assert normalize_state("N5 厌烦/抗拒（软拒绝）") == "displeased"
    assert normalize_state("N1 挫败/升级（要投诉）") == "displeased"
    assert normalize_state("N5 厌烦/敷衍") == "displeased"
    assert normalize_state("P3 感激/满意") == "cooperative"
    assert normalize_state("P4 好奇/感兴趣") == "cooperative"


def test_normalize_state_embedded_legacy_code():
    """句中嵌入旧码的自由文本（oracle_state 口径）同样归一。"""
    assert normalize_state("用户处于P2_relaxed（P2 轻松/随和）状态，动态类型 all_positive") == "cooperative"
    assert normalize_state("用户处于N6_suspicious状态") == "doubtful"


def test_normalize_state_unknown():
    assert normalize_state("未知状态") is None
    assert normalize_state("") is None
    assert normalize_state(None) is None


def test_shared_action_labels():
    from listen2serve.domains.base import ACTION_TABLE_ORDER, action_table

    table = action_table()
    for state in ACTION_TABLE_ORDER:
        assert f"听出【{STATE_LABELS[state]}】" in table


def test_render_policy_prompt_turn_budget_pace_line():
    domain = get_domain("collection")
    prompt = domain.render_policy_prompt(turn_budget={"soft": 6, "hard": 10})
    assert "本通电话预计约 6 轮内完成主要流程，请合理推进不要拖沓，并为收尾轮留出余量。" in prompt
    assert "本通电话预计约" not in domain.render_policy_prompt()
    assert "不跳步" not in prompt
