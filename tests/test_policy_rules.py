"""Plan I（v6.0）回归测试：三域 policy_rules 新 6 态全覆盖 + 状态键归一化兼容新旧格式。

防回归点：
1. 任何域的 policy_rules 键集合必须覆盖全部 6 个 canonical 状态；
2. normalize_state 同时兼容新 6 态码、旧 11 态码与中文标签（旧 run 重评）；
3. 被测 prompt 的「用户状态应对策略」段标题由 STATE_LABELS 派生（单一来源）；
4. STATE_LABELS 与 emotion.STATE_EMOTION 同码。
"""

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


def test_render_policy_prompt_labels_from_state_labels():
    """被测 prompt「用户状态应对策略」段标题由 STATE_LABELS 派生（6 态全覆盖，两版本各自措辞）。"""
    heads = {"v6": "## 当用户表现出[{}]时：", "v7": "## 当你从客户声音中听出[{}]时："}
    for role in available_roles():
        domain = get_domain(role)
        for variant, head in heads.items():
            prompt = domain.render_policy_prompt(variant=variant)
            assert "# 用户状态应对策略" in prompt
            for label in STATE_LABELS.values():
                assert head.format(label) in prompt
    # v7 把「状态判据是声音」写进段标题（听→做任务面的入口）
    v7 = get_domain("collection").render_policy_prompt(variant="v7")
    assert "# 用户状态应对策略（状态以你听到的声音为主要判据；本节优先于上方业务流程）" in v7


def test_render_policy_prompt_turn_budget_pace_line():
    """Plan I：turn_budget.soft 存在时末尾追加节奏引导行；缺省不输出。

    v7 改写文案：删去与「状态优先、流程可跳转可回补」矛盾的「不跳步」，改为要求为收尾轮留余量。
    """
    domain = get_domain("collection")
    expected = {
        "v6": "本通电话预计约 6 轮内完成主要流程，请合理推进，既不拖沓也不跳步。",
        "v7": "本通电话预计约 6 轮内完成主要流程，请合理推进不要拖沓，并为收尾轮留出余量。",
    }
    for variant, line in expected.items():
        prompt = domain.render_policy_prompt(turn_budget={"soft": 6, "hard": 10}, variant=variant)
        assert line in prompt
        assert "本通电话预计约" not in domain.render_policy_prompt(variant=variant)
    assert "不跳步" not in domain.render_policy_prompt(turn_budget={"soft": 6}, variant="v7")
