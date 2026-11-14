"""The critical-turn and dialogue judges share the published service contract."""

from listen2serve.domains import get_domain
from listen2serve.domains.base import (
    STATE_ACTIONS,
    judge_action_table,
    render_customer_profile,
    render_judge_contract,
)
from listen2serve.evaluation.flow_judge import _render_rules as flow_rules
from listen2serve.evaluation.policy_judge import _render_rules as keyturn_rules
from listen2serve.evaluation.prompts import (
    KEYTURN_ACTION_CHEATSHEET,
    KEYTURN_JUDGE_PROMPT,
    build_judge_meta,
    prompt_hash,
)

SCENARIO = {
    "role": "collection",
    "oracle_state": "displeased",
    "agent_policy": {
        "role_description": "催收客服",
        "communication_style": "坚定、冷静",
        "required_actions": ["确认身份", "促成还款承诺"],
        "forbidden_actions": ["威胁辱骂"],
        "allowed_actions": ["分期方案"],
        "voice_requirements": "语速中等，坚定不攻击",
    },
    "expected_key_behavior": "回应困难并推进还款安排",
}


def test_contract_includes_scenario_requirements_and_current_state():
    text = render_judge_contract(get_domain("collection"), SCENARIO, current_state="displeased")
    for expected in (
        "催收客服",
        "坚定、冷静",
        "必需动作：确认身份；促成还款承诺",
        "禁止动作：威胁辱骂",
        "允许动作：分期方案",
        "语速中等，坚定不攻击",
        "回应困难并推进还款安排",
        "←当前状态",
    ):
        assert expected in text


def test_both_judges_use_the_shared_contract_and_same_customer_profile():
    domain = get_domain("collection")
    contract = render_judge_contract(domain, SCENARIO)
    assert keyturn_rules(domain, SCENARIO) == flow_rules(domain, SCENARIO) == contract
    import json
    from pathlib import Path

    scenario = json.loads(Path("data/benchmark/scenarios.jsonl").read_text().splitlines()[0])
    domain = get_domain(scenario["role"])
    profile = render_customer_profile(domain, scenario["db_seed"])
    assert profile
    assert keyturn_rules(domain, scenario).endswith(profile)
    assert flow_rules(domain, scenario).endswith(profile)


def test_action_table_is_injected_from_single_source():
    assert KEYTURN_ACTION_CHEATSHEET.strip() in KEYTURN_JUDGE_PROMPT
    assert "{action_cheatsheet}" not in KEYTURN_JUDGE_PROMPT
    assert judge_action_table().strip() in KEYTURN_ACTION_CHEATSHEET.replace("     ", "")
    for row in STATE_ACTIONS.values():
        assert row["opening"] in KEYTURN_JUDGE_PROMPT
    assert "判据优先级" in KEYTURN_JUDGE_PROMPT
    assert "未做不单独扣分" in KEYTURN_JUDGE_PROMPT


def test_judge_metadata_identifies_models_and_prompt_contents():
    meta = build_judge_meta("google/gemini-3.8-flash", "openai/audio", profile_injected=True)
    assert meta["protocol"] == "multi-turn/1"
    assert meta["keyturn"]["model"] == meta["flow"]["model"] == "google/gemini-3.8-flash"
    assert meta["voice"]["model"] == "openai/audio"
    assert meta["keyturn"]["profile_injected"] and meta["flow"]["profile_injected"]
    assert "profile_injected" not in meta["voice"]
    assert meta["keyturn"]["prompt_hash"] == prompt_hash(KEYTURN_JUDGE_PROMPT)
    assert prompt_hash("a") != prompt_hash("b")


def test_every_published_forbidden_action_is_in_the_judge_contract():
    """Coverage validation uses exactly the unfiltered scenario-action section."""
    import json
    from pathlib import Path

    for line in Path("data/benchmark/scenarios.jsonl").read_text().splitlines():
        scenario = json.loads(line)
        contract = render_judge_contract(get_domain(scenario["role"]), scenario)
        section = contract.split("## 场景动作约束\n", 1)[1].split("\n## ", 1)[0]
        forbidden = scenario["agent_policy"].get("forbidden_actions") or []
        assert "禁止动作：" + "；".join(forbidden) in section
        assert not any("字数" in action or "复读" in action for action in forbidden)
