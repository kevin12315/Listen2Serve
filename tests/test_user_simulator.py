"""骨架驱动用户模拟器契约测试。

防回归点：
1. 输出协议：紧凑 JSON、字段硬校验（text 10–60 字/state 属本场景候选集）；
2. 唯一关键轮状态机：首次 true 锁存、重复降级、soft 告警、hard-2 强制
   （forced_key_turn）、end_call 须在关键事件完成后；
3. TTS instruct 表驱动：按 base_scenario_id × state 查预生成表，
   查不到即 raise（不回落）；同 base 同 state 两条件逐字一致；标签四重校验与
   text/text_tts 分离；
4. 词表防线：T3 全轮 + T1 非关键轮零命中；T1 关键轮正向命中（标记不阻断）；
5. 预算与终止：end_call 后 next_turn 返回 None（user_closed）；
6. 打断：peek_interrupt 预生成草案（无状态副作用）。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.tts_instructions import load_table
from listen2serve.runtime.tts_tags import STATE_TAG_ALLOWED
from listen2serve.runtime.user_simulator import (
    MAX_UTTERANCE_CHARS,
    MIN_UTTERANCE_CHARS,
    OUTBOUND_OPENING_POOL,
    UserSimulator,
    _instruct_from_control,
    _parse_protocol,
    normalize_leakage_level,
)

# 强制标签从声音表常量派生，避免写死标签名。
_DISPLEASED_TAG = STATE_TAG_ALLOWED["displeased"]


def _proto(text: str = "喂你好，我就是本人，你说吧，什么事？", **over) -> str:
    d = {
        "text": text,
        "state": "neutral",
        "is_key_turn": False,
        "end_call": False,
        "covered": [],
        "interrupt": False,
    }
    d.update(over)
    return json.dumps(d, ensure_ascii=False)


def _script(**over) -> dict:
    us = {
        "identity": "张三，35 岁上班族，有一笔 3600 元欠款",
        "user_facts": {"planned_amount": 1800, "planned_date": "最迟下周五"},
        "user_goal": "核对清楚后给出可行还款安排",
        "mission": [
            "接听电话，确认自己是本人",
            "弄清来意与欠款情况",
            "【关键事件】摊牌还款能力：先还九千其余宽限",
            "确认还款渠道与到账时间",
            "达成约定后道别",
        ],
        "key_event": {
            "description": "摊牌还款能力：先还九千，其余宽限",
            "state": "cooperative",
            "trigger_hint": "客服给出具体还款要求之后",
            "interrupt": False,
        },
        "disclosure_plan": {"planned_amount": "客服问到还款能力前不主动说"},
        "turn_budget": {"soft": 5, "hard": 9},
        "persona_style": ["口头禅「这样啊」"],
    }
    us.update(over)
    return us


def _sim(user_script: dict | None = None, leakage: str = "T3", **kw):
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = MagicMock(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sid = kw.pop("scenario_id", "T")
    us = user_script or _script()
    # 合成场景不在真表里：注入同构条目（仍过 validate_entry）
    kw.setdefault(
        "instruction_entry", fake_instruction_entry(str((us.get("key_event") or {}).get("state") or ""))
    )
    sim = UserSimulator(
        us,
        llm_gateway=llm,
        tts_gateway=tts,
        llm_model="stub-judge-model",
        role="collection",
        dynamics="all_positive",
        leakage_label=leakage,
        user_gender="male",
        scenario_id=sid,
        seed=42,
        **kw,
    )
    return sim, llm


def _reply(llm, **over):
    llm.chat.return_value = MagicMock(text=_proto(**over), finish_reason="stop")


# ---- 1. 协议解析 ----


def test_parse_protocol_lenient():
    assert _parse_protocol('```json\n{"text":"x","state":"neutral"}\n```')["state"] == "neutral"
    assert _parse_protocol('好的：{"text":"x"} 谢谢')["text"] == "x"
    assert _parse_protocol("不是 JSON") is None
    assert _parse_protocol("") is None


def test_normalize_leakage_level():
    assert normalize_leakage_level("explicit") == "T1"
    assert normalize_leakage_level("T3") == "T3"
    assert normalize_leakage_level("unknown") == ""


# ---- 2. 骨架构件与 prompt ----


def test_skeleton_required_fields():
    with pytest.raises(ValueError):
        UserSimulator({"key_event": {"description": "x", "state": "cooperative"}}, MagicMock(), MagicMock())
    with pytest.raises(ValueError):
        UserSimulator(
            _script(key_event={"description": "x", "state": "N5_resistant"}), MagicMock(), MagicMock()
        )


def test_prompt_sections_and_stable_prefix():
    sim, _ = _sim(leakage="T1")  # T1 口径含关键事件参考词注入
    m1 = sim._build_messages(1, "", forced=False)
    m2 = sim._build_messages(1, "", forced=False)
    assert m1 == m2  # 稳定前缀逐字节可复现（缓存友好）
    user = m1[1]["content"]
    for section in (
        "【身份】",
        "【说话风格】",
        "【目标】",
        "【事实锚】",
        "【用户本人掌握的事实】",
        "【披露时机】",
        "【剧情要点】",
        "【关键事件】",
        "【透明度约束】",
        "【轮次预算】",
        "【完整对话历史】",
    ):
        assert section in user, section
    assert "口头禅" in user and "至多 1 次" in user  # 口头禅封顶条款保留
    assert "参考词" in user  # T1 抽样词注入（关键事件轮正向要求）
    sys_prompt = m1[0]["content"]
    assert "我都听你的安排" in sys_prompt  # 自述式负例
    assert "紧凑单行 JSON" in sys_prompt


def test_budget_warning_lines():
    sim, _ = _sim()
    for _ in range(5):
        sim._turn_count += 1  # 模拟已到 soft
    m = sim._build_messages(6, "", forced=False)
    assert "最迟下一轮必须完成【关键事件】" in m[1]["content"]
    # hard-2 强制条款
    sim._turn_count = 6
    m = sim._build_messages(7, "", forced=True)
    assert "本轮必须完成【关键事件】" in m[1]["content"]


# ---- 3. 唯一关键轮状态机 ----


def test_key_turn_latch_and_downgrade():
    sim, llm = _sim()
    _reply(llm)
    r1 = sim.next_turn("您好，是张三先生吗？")
    assert not r1.is_key_turn
    _reply(
        llm,
        text="行，下月十号前我先还九千，其余宽限几天。",
        state="cooperative",
        is_key_turn=True,
        covered=[2, 3],
    )
    r2 = sim.next_turn("您打算什么时候处理？")
    assert r2.is_key_turn and sim.key_completed
    _reply(llm, text="嗯，那渠道和到账时间你说一下吧。", state="cooperative", is_key_turn=True)
    r3 = sim.next_turn("好的我帮您登记。")
    assert not r3.is_key_turn  # 重复声明被降级
    assert sim.stats["key_downgrade"] == 1


def test_forced_key_turn_at_hard_minus_2():
    sim, llm = _sim()  # hard=9 → n=7 强制
    for i in range(6):
        _reply(llm, text=f"嗯，这个情况我知道了，你先说说第{i}点吧。")
        sim.next_turn(f"客服话术{i}，请问还有什么问题吗？")
    _reply(llm, text="行吧，那就先按这个方案走着看吧。")
    r = sim.next_turn("您看这样处理可以吗？")
    assert r.is_key_turn and r.forced_key_turn
    assert r.state == "cooperative"  # 强制置 key_event.state
    assert sim.stats["forced_key_turn"] == 1


def test_end_call_requires_key_event():
    sim, llm = _sim()
    _reply(llm, end_call=True)
    r = sim.next_turn("您好，是张三先生吗？")
    assert not r.end_call  # 关键事件未完成 → 拒绝并置 false
    assert sim.stats["end_call_rejected"] == 1
    assert not sim._ended


def test_end_call_closes_call():
    sim, llm = _sim()
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。", state="cooperative", is_key_turn=True)
    sim.next_turn("您打算怎么处理？")
    _reply(llm, text="好的，那就这么说定了，麻烦你了，再见。", end_call=True)
    r = sim.next_turn("我帮您登记好了。")
    assert r.end_call and sim.termination_reason == "user_closed"
    assert sim.next_turn("再见") is None


def test_state_mismatch_overridden_with_data():
    sim, llm = _sim()
    # 关键轮 state 报错 → 重试仍报（mock 不变）→ 以数据为准覆盖并记 state_mismatch
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。", state="displeased", is_key_turn=True)
    r = sim.next_turn("您打算怎么处理？")
    assert r.is_key_turn and r.state == "cooperative"
    assert r.state_mismatch and sim.stats["state_mismatch"] == 1


def test_hard_protocol_failure_raises():
    sim, llm = _sim()
    # 两次都输出非法 JSON → 任务级失败（评测通路无兜底）
    llm.chat.return_value = MagicMock(text="我不是 JSON", finish_reason="stop")
    with pytest.raises(RuntimeError):
        sim.next_turn("您好，是张三先生吗？")


def test_length_retry_then_accept():
    sim, llm = _sim()
    short = MagicMock(text=_proto(text="嗯。"), finish_reason="stop")
    ok = MagicMock(text=_proto(), finish_reason="stop")
    llm.chat.side_effect = [short, ok]
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.text == "喂你好，我就是本人，你说吧，什么事？"
    assert sim.stats["retries"] == 1


def test_instruct_lookup_verbatim_from_table():
    """查表命中 → instruct 与表内文本逐字一致（运行期不再拼任何韵律文本）。"""
    entry = fake_instruction_entry("cooperative")
    sim, llm = _sim(instruction_entry=entry)
    _reply(llm)
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.tts_control == entry["non_key"]["neutral"]
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。", state="cooperative", is_key_turn=True)
    r2 = sim.next_turn("您打算怎么处理？")
    assert r2.tts_control == entry["key"]  # 关键轮固定取 key 条（不分透明度）


def test_instruct_real_table_lookup():
    """真实表条目：base_scenario_id 由 scenario_id 剥尾缀推出，取值逐字来自表。"""
    table = load_table()
    base = "COL-ALL-01"
    entry = table.entry(base)
    sc = _script(
        key_event={
            "description": "摊牌",
            "state": entry["key_state"],
            "trigger_hint": "x",
            "interrupt": False,
        }
    )
    sim, llm = _sim(user_script=sc, scenario_id=f"{base}-T3", instruction_entry=None)
    assert sim.base_scenario_id == base
    _reply(llm)
    assert sim.next_turn("您好，是张三先生吗？").tts_control == entry["non_key"]["neutral"]


def test_instruct_reaches_tts_call():
    """instruction 确已下发到 TTS：比对传给 synthesize 的 style 实参（§0.15 i 第 1 项）。

    只验落盘的 tts_control 是不够的 —— 历史上真出过「落盘有值、实际没发出去」的缺陷
    （字段名写错即被后端静默忽略，音频与不带指令逐字节相同），所以这里直接握
    网关入参：style 必须是表内文本包上「用…的语气」，且与落盘值同源。
    """
    entry = fake_instruction_entry("displeased")
    sc = _script(
        key_event={"description": "摊牌", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    sim, llm = _sim(user_script=sc, instruction_entry=entry)
    _reply(llm)
    r = sim.next_turn("您好，是张三先生吗？")
    kwargs = sim.tts.synthesize.call_args.kwargs
    assert kwargs["style"] == f"用{entry['non_key']['neutral']}的语气"
    assert kwargs["style"] == f"用{r.tts_control}的语气"  # 与落盘值同源
    # 关键轮同样：下发的必须是表内 key 条，不是非关键轮那条
    _reply(llm, text="我这边真的没法接受这个安排。", state="displeased", is_key_turn=True)
    sim.next_turn("您看这样安排行吗？")
    assert sim.tts.synthesize.call_args.kwargs["style"] == f"用{entry['key']}的语气"


def test_instruct_missing_state_raises():
    """查表缺 state → raise（不回落模板：回落会让声音与 state 无关且无人察觉）。"""
    sim, _ = _sim(instruction_entry=fake_instruction_entry("cooperative", states=["neutral", "cooperative"]))
    assert sim.candidate_states == ("neutral", "cooperative")
    with pytest.raises(ValueError, match="instruction 表缺 state"):
        sim._assemble_instruct("urgent", is_key=False)


def test_instruct_table_missing_file_raises_at_init(tmp_path):
    """表文件缺失 → 构造期就 raise（不等到第一轮合成才发现）。"""
    with pytest.raises(ValueError, match="instruction 表不存在"):
        UserSimulator(
            _script(),
            MagicMock(),
            MagicMock(),
            scenario_id="COL-ALL-01-T3",
            instruction_table_path=tmp_path / "nope.json",
        )


def test_instruct_from_control_contract_raises():
    """instruct 契约违约 → raise（原 warning 的反向测试：表坏了必须当场炸）。"""
    ok = _instruct_from_control("neutral", "一位客户，语气平稳，语速中等，音量正常")
    assert ok.startswith("用") and ok.endswith("的语气")
    with pytest.raises(ValueError, match="不符契约"):
        _instruct_from_control("neutral", "")
    with pytest.raises(ValueError, match="不符契约"):
        _instruct_from_control("neutral", "一位客户，语气平稳")  # 无语速/音量词
    with pytest.raises(ValueError, match="硬上限"):
        _instruct_from_control("neutral", "语速中等，音量正常" + "啊" * 80)


def test_instruct_flat_across_levels():
    """韵律条件一致：同 base 同 state，T1/T3 的 instruct 逐字一致。

    表的索引键不含 -T1/-T3/-T3，这条不变量由数据结构保证；本测试守住接线不跑偏。
    """
    non_key, key = [], []
    for level in ("T1", "T3"):
        sim, llm = _sim(leakage=level)
        _reply(llm)
        non_key.append(sim.next_turn("您好，是张三先生吗？").tts_control)
        _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。", state="cooperative", is_key_turn=True)
        key.append(sim.next_turn("您打算怎么处理？").tts_control)
    assert len(set(non_key)) == 1 and len(set(key)) == 1


def test_prosody_arm_neutral_is_state_and_keyturn_invariant():
    """中性臂：instruct 恒等于 non_key[neutral]，与本轮 state / is_key 全无关。

    这是 A/C 两格「韵律恒中性」的定义 —— 只有它能被主动产生，不再寄生于
    2026-08-26 那批 instruction 未下发的 bug。
    """
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"])
    sc = _script(
        key_event={"description": "拒绝安排", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    sim, _ = _sim(user_script=sc, instruction_entry=entry, prosody_arm="neutral")
    neutral = entry["non_key"]["neutral"]
    assert sim._assemble_instruct("neutral", is_key=False) == neutral
    assert sim._assemble_instruct("displeased", is_key=False) == neutral  # 非中性 state 也拉平
    assert sim._assemble_instruct("displeased", is_key=True) == neutral  # 关键轮不取 key 条


def test_prosody_arm_neutral_drops_forced_tag():
    """中性臂：不附加任何强制标签（[angry]/[whispers] 本身即韵律，留着会污成半中性）。"""
    entry = fake_instruction_entry(
        "displeased",
        states=["neutral", "displeased"],
        tags={"displeased": _DISPLEASED_TAG},
        key_tag=_DISPLEASED_TAG,
    )
    sc = _script(
        key_event={"description": "拒绝安排", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    sim, llm = _sim(user_script=sc, instruction_entry=entry, prosody_arm="neutral")
    assert sim._forced_tag("displeased", is_key=False) == ""
    assert sim._forced_tag("displeased", is_key=True) == ""
    _reply(llm, text="这个安排我接受不了，得再商量商量。", state="displeased", is_key_turn=True)
    r = sim.next_turn("您看这样安排行吗？")
    assert r.text_tts == r.text  # 无标签，两字段逐字一致
    assert r.text_tts == "这个安排我接受不了，得再商量商量。"


@pytest.mark.parametrize("prosody_arm", ["neutral", "state"])
@pytest.mark.parametrize("state,tag", [("displeased", "angry"), ("doubtful", "curious")])
def test_model_state_tags_follow_prosody_arm_without_changing_key_state(prosody_arm, state, tag):
    entry = fake_instruction_entry(state, states=["neutral", state], tags={state: tag}, key_tag=tag)
    script = _script(
        key_event={"description": "核对业务", "state": state, "trigger_hint": "x", "interrupt": False}
    )
    sim, llm = _sim(user_script=script, instruction_entry=entry, prosody_arm=prosody_arm)
    text = "这笔业务的金额我先核对一下再说。"
    _reply(llm, text=f"[{tag}]{text}", state=state, is_key_turn=True)

    result = sim.next_turn("请您先核对一下这笔业务。")

    assert result.is_key_turn is True
    assert result.state == state
    assert result.text == text
    assert result.llm_debug["tag_raw"] == [tag]
    if prosody_arm == "neutral":
        assert result.text_tts == text
        assert result.tts_control == entry["non_key"]["neutral"]
        assert result.llm_debug["tag_applied"] == ""
        assert sim.stats["tag_stripped"] == 1
    else:
        assert result.text_tts == f"[{tag}]{text}"
        assert result.tts_control == entry["key"]
        assert result.llm_debug["tag_applied"] == tag
    assert sim.tts.synthesize.call_args.args[0] == result.text_tts
    assert sim.tts.synthesize.call_args.kwargs["style"] == f"用{result.tts_control}的语气"


def test_prosody_arm_default_is_state():
    """默认臂 = state：不传 prosody_arm 时行为不变（关键轮仍取 key 条、带强制标签）。"""
    entry = fake_instruction_entry(
        "displeased",
        states=["neutral", "displeased"],
        tags={"displeased": _DISPLEASED_TAG},
        key_tag=_DISPLEASED_TAG,
    )
    sc = _script(
        key_event={"description": "拒绝安排", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    sim, _ = _sim(user_script=sc, instruction_entry=entry)
    assert sim.prosody_arm == "state"
    assert sim._assemble_instruct("displeased", is_key=True) == entry["key"]
    assert sim._forced_tag("displeased", is_key=True) == _DISPLEASED_TAG


def test_prosody_arm_invalid_raises_at_init():
    """非法臂名 → 构造期 raise（只接 state / neutral）。"""
    with pytest.raises(ValueError, match="prosody_arm"):
        _sim(prosody_arm="middle")


def test_sim_thinking_default_off_reaches_gateway():
    """默认必须仍下发 thinking=False。

    这既是 §10.6 的成本红线，也是对照有效性的前提：已跑完的步骤 4 两臂都是默认态，
    只有默认确实为 False，它们才能当 thinking=OFF 的对照臂用。
    """
    sim, llm = _sim()
    assert sim.sim_thinking is False
    _reply(llm)
    sim.next_turn("您好，是张三先生吗？")
    assert llm.chat.call_args.kwargs["thinking"] is False


def test_sim_thinking_on_reaches_gateway():
    """开关必须真的传到 llm.chat —— 只存在对象上不算开（那样会跑出一臂假 ON 数据）。"""
    sim, llm = _sim(sim_thinking=True)
    assert sim.sim_thinking is True
    _reply(llm)
    sim.next_turn("您好，是张三先生吗？")
    assert llm.chat.call_args.kwargs["thinking"] is True


def test_forced_tag_applied_and_text_split():
    """表驱动强制标签：text_tts 带句首标签、text 干净（评委/ASR 只看 text）。

    §0.12f（当时实测用 `[reluctantly]`）：标签会让后端在句首带出一声叹词，音频与文本
    天然不逐字对应——这正是两个字段必须分离的原因，文本类指标不该为此波动。
    """
    entry = fake_instruction_entry(
        "displeased",
        states=["neutral", "displeased"],
        tags={"displeased": _DISPLEASED_TAG},
        key_tag=_DISPLEASED_TAG,
    )
    sc = _script(
        key_event={"description": "拒绝安排", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    sim, llm = _sim(user_script=sc, instruction_entry=entry)
    _reply(llm, text="这个安排我接受不了，得再商量商量。", state="displeased")
    r = sim.next_turn("您看这样安排行吗？")
    assert r.text == "这个安排我接受不了，得再商量商量。"  # 干净台词
    assert r.text_tts == f"[{_DISPLEASED_TAG}]这个安排我接受不了，得再商量商量。"
    assert r.llm_debug["tag_applied"] == _DISPLEASED_TAG
    assert sim.stats["tag_stripped"] == 0


def test_tag_policy_four_checks():
    """标签四重校验：非白名单/非允许集/非句首/富语言未启用各一例 → 一律剥离并计数。

    非允许集那一例用的是**别态的合法标签**（`[whispers]` 是 hushed 的允许标签）：它过得了
    白名单但过不了本 base 的允许集，正是「一态一标签」落到运行期的那条防线。
    """
    entry = fake_instruction_entry(
        "displeased",
        states=["neutral", "displeased"],
        tags={"displeased": _DISPLEASED_TAG},
        key_tag=_DISPLEASED_TAG,
    )
    sc = _script(
        key_event={"description": "拒绝安排", "state": "displeased", "trigger_hint": "x", "interrupt": False}
    )
    cases = {
        "[激动]我这边真的没法接受这个安排。": "not_whitelisted",
        "[whispers]我这边真的没法接受这个安排。": "not_in_candidate_set",
        f"我这边[{_DISPLEASED_TAG}]真的没法接受这个安排。": "not_at_head",
        "[sighing]我这边真的没法接受这个安排。": "rich_disabled",
    }
    for raw, reason in cases.items():
        sim, llm = _sim(user_script=sc, instruction_entry=entry)
        _reply(llm, text=raw, state="neutral")
        r = sim.next_turn("您看这样安排行吗？")
        assert reason in " ".join(r.llm_debug["tag_stripped_reason"]), raw
        assert "[" not in r.text and "[" not in r.text_tts  # neutral 无强制标签
        assert sim.stats["tag_stripped"] == 1
    # 数量上限：强制标签已占句首，LLM 自带的同名标签一律被「非句首」挡下 —— 位置规则
    # 蕴含了数量规则（offset 0 只可能有一个标签），故 exceeds_max_count 在本路径不可达。
    sim, llm = _sim(user_script=sc, instruction_entry=entry)
    _reply(
        llm, text=f"[{_DISPLEASED_TAG}][{_DISPLEASED_TAG}]我这边真的没法接受这个安排。", state="displeased"
    )
    r = sim.next_turn("您看这样安排行吗？")
    assert r.text_tts.count(f"[{_DISPLEASED_TAG}]") == 1
    assert "[" not in r.text
    assert [x.split(":")[1] for x in r.llm_debug["tag_stripped_reason"]] == ["not_at_head", "not_at_head"]


def test_runtime_no_emotion_template_reference():
    """运行期路径 0 引用 emotion.tts_control_for（模板拼装整体退役）。"""
    from pathlib import Path

    from listen2serve.runtime import user_simulator as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "tts_control_for" not in src
    for retired in (
        "MAX_TONE_CHARS",
        "RATE_SLOT",
        "VOLUME_SLOT",
        "_KEY_TURN_TONE_BY_STATE",
        "_INTENSITY_NAME",
        "key_intensity",
    ):
        assert retired not in src, retired


def test_opening_instruct_independent_of_level():
    """开场轮 instruct 不再随 leakage_label 变化（固定 neutral 查表）。"""
    controls = []
    for level in ("T1", "T3"):
        sim, _ = _sim(leakage=level, role_type="outbound_pressure", scenario_id="COL-ALL-01")
        t1 = sim.next_turn("")
        controls.append(t1.tts_control)
        assert t1.state == "neutral"
    assert len(set(controls)) == 1


def test_state_drift_counted_not_overridden():
    """mission 声明的 state 与 LLM 自报不符 → 只记 state_drift，不覆盖（§4.4）。"""
    sc = _script(
        mission=[
            {"beat": "接听电话，确认自己是本人", "state": "neutral"},
            {"beat": "弄清来意与欠款情况", "state": "doubtful"},
            {"beat": "【关键事件】摊牌还款能力", "state": "cooperative", "key_event": True},
        ]
    )
    sim, llm = _sim(user_script=sc)
    _reply(llm, state="neutral", covered=[2])  # mission 第 2 条要 doubtful
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.state == "neutral" and sim.stats["state_drift"] == 1
    assert r.llm_debug["state_expected"] == "doubtful"


# ---- 5. 词表防线 ----


def test_vocab_neutral_zero_hit_defense():
    sim, llm = _sim(leakage="T3")
    # 首试命中情绪词 → 反馈重试；重试仍命中（mock 不变）→ 保留并标记
    _reply(llm, text="我很烦你们老是打电话过来催我还款。")
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.vocab_violation and sim.stats["vocab_violation"] == 1
    assert sim.stats["retries"] == 1


def test_vocab_t1_key_positive_check():
    sim, llm = _sim(leakage="T1")
    # 关键轮未命中 cooperative 子表词 → 重试后标记（不阻断）
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。", state="cooperative", is_key_turn=True)
    r = sim.next_turn("您打算怎么处理？")
    assert r.vocab_violation
    # 命中子表词 → 合规
    sim2, llm2 = _sim(leakage="T1")
    _reply(llm2, text="没问题，我按你说的来，下月十号前先还九千。", state="cooperative", is_key_turn=True)
    r2 = sim2.next_turn("您打算怎么处理？")
    assert not r2.vocab_violation


# ---- 6. 打断预生成 ----


def test_peek_interrupt_draft_no_side_effect():
    sc = _script(
        key_event={"description": "摊牌", "state": "cooperative", "trigger_hint": "x", "interrupt": True}
    )
    sim, llm = _sim(user_script=sc)
    _reply(
        llm,
        text="你等等，我先说，下月十号前我先还九千行不行？",
        state="cooperative",
        is_key_turn=True,
        interrupt=True,
    )
    assert sim.peek_interrupt("您先别急，听我把这个情况解释一下") is True
    assert sim.peek_interrupt("您先别急，听我把这个情况解释一下") is True
    # 无副作用：状态机未推进
    assert not sim.key_completed and sim._turn_count == 0
    # interrupt=False 场景不预生成
    sim2, llm2 = _sim()
    _reply(llm2)
    assert sim2.peek_interrupt("您好，请问是张三先生本人吗？") is False
    assert llm2.chat.call_count == 0


def test_char_limits_constants():
    assert MIN_UTTERANCE_CHARS == 10
    assert MAX_UTTERANCE_CHARS == 60


def test_outbound_opening_passive_no_llm():
    """外呼首轮：应答池被动接听，不走 LLM、不注入 mission（消除自答伪影）。"""
    sim, llm = _sim(role_type="outbound_pressure", scenario_id="COL-ALL-01")
    t1 = sim.next_turn("")
    assert t1.text in OUTBOUND_OPENING_POOL
    assert t1.state == "neutral" and t1.is_key_turn is False
    assert not t1.text.startswith("对") and "本人" not in t1.text  # 无自答身份
    llm.chat.assert_not_called()
    assert sim.stats["outbound_opening"] == 1
    assert sim._turn_count == 1


def test_outbound_opening_deterministic():
    """crc32 确定性：同 scenario_id 同句；不同 id 可不同（不强制）。"""
    sim1, _ = _sim(role_type="outbound_pressure", scenario_id="COL-ALL-01")
    sim2, _ = _sim(role_type="outbound_pressure", scenario_id="COL-ALL-01")
    assert sim1.next_turn("").text == sim2.next_turn("").text


def test_outbound_second_turn_elicit():
    """mission 第 2 轮起生效：第 2 轮正常走 LLM elicit。"""
    sim, llm = _sim(role_type="outbound_negotiation", scenario_id="MAR-ALL-01")
    sim.next_turn("")
    _reply(llm)
    t2 = sim.next_turn("您好，请问是本人吗？")
    llm.chat.assert_called()
    assert sim.stats["outbound_opening"] == 1  # 仅首轮计入
    assert t2.text  # LLM 演绎轮


def test_inbound_opening_unchanged():
    """hotline（inbound）不受影响：首轮仍走 LLM。"""
    sim, llm = _sim(role_type="inbound_care")
    _reply(llm)
    sim.next_turn("")
    llm.chat.assert_called()
    assert sim.stats["outbound_opening"] == 0


def test_prompt_keeps_full_history_and_current_progress():
    sim, _ = _sim()
    sim._history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"完整历史第{i}条"} for i in range(12)
    ]
    user = sim._build_messages(7, "您看具体哪天能还呢？", forced=False)[1]["content"]
    assert "【完整对话历史】" in user
    for message in sim._history:
        assert message["content"] in user
    assert "【进度】" in user
    assert "【客服上轮回复】您看具体哪天能还呢？" in user


def test_prompt_keeps_role_and_facts_boundaries():
    sim, _ = _sim()
    system, user = [m["content"] for m in sim._build_messages(1, "", forced=False)]
    assert "绝不能说客服的台词" in system
    assert "【事实锚使用边界】" in user
    assert "【用户本人掌握的事实】" in user
    assert "【披露时机】" in user
    assert "【透明度约束】" in user
    assert "【风格示范】" in user


def test_completed_key_event_and_soft_budget_request_closing():
    sim, _ = _sim()
    sim._key_turn_no = 2
    user = sim._build_messages(6, "还有什么需要确认？", forced=False)[1]["content"]
    assert "已在第 2 轮完成" in user
    assert "关键事件已完成不等于事情已办完" in user
    assert "推进仍适用的剧情" in user
    assert "事已办完，请尽快道别收尾" not in user
    assert "本轮必须完成【关键事件】" not in user
