"""Plan I（v6.0）骨架驱动用户模拟器契约测试。

防回归点：
1. 输出协议：紧凑 JSON、字段硬校验（text 10–60 字/state 属本场景候选集）；
2. 唯一关键轮状态机：首次 true 锁存、重复降级、soft 告警、hard-2 强制
   （forced_key_turn）、end_call 须在关键事件完成后；
3. TTS instruct 表驱动（Plan L §4.1）：按 base_scenario_id × state 查预生成表，
   查不到即 raise（不回落）；同 base 同 state 三档逐字一致；标签四重校验与
   text/text_tts 分离；
4. 词表防线：T2/T3 全轮 + T1 非关键轮零命中；T1 关键轮正向命中（标记不阻断）；
5. 预算与终止：end_call 后 next_turn 返回 None（user_closed）；
6. 打断：peek_interrupt 预生成草案（无状态副作用）。
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from conftest import fake_instruction_entry

from listen2serve.runtime.tts_tags import STATE_TAG_ALLOWED
from listen2serve.runtime.tts_instructions import load_table
from listen2serve.runtime.user_simulator import (
    MAX_UTTERANCE_CHARS,
    MIN_UTTERANCE_CHARS,
    OUTBOUND_OPENING_POOL,
    UserSimulator,
    _instruct_from_control,
    _parse_protocol,
    normalize_leakage_level,
)

# 强制标签的取值一律从代码常量派生，不逐字写死：Step2（v6.4）把 displeased 从
# `reluctantly` 换到 `angry`，写死标签名的测试会集体失效，而失效方式看起来像产品 bug。
_DISPLEASED_TAG = STATE_TAG_ALLOWED["displeased"]


def _proto(text: str = "喂你好，我就是本人，你说吧，什么事？", **over) -> str:
    d = {
        "text": text, "state": "neutral", "is_key_turn": False,
        "end_call": False, "covered": [], "interrupt": False,
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
            "trigger_hint": "客服给出具体还款要求之后", "interrupt": False,
        },
        "disclosure_plan": {"planned_amount": "客服问到还款能力前不主动说"},
        "turn_budget": {"soft": 5, "hard": 9},
        "persona_style": ["口头禅「这样啊」"],
    }
    us.update(over)
    return us


def _sim(user_script: dict | None = None, leakage: str = "T2", **kw):
    llm, tts = MagicMock(), MagicMock()
    tts.synthesize.return_value = MagicMock(audio=b"\x00\x00" * 1600, sample_rate=16000)
    sid = kw.pop("scenario_id", "T")
    us = user_script or _script()
    # 合成场景不在真表里：注入同构条目（仍过 validate_entry）
    kw.setdefault("instruction_entry",
                  fake_instruction_entry(str((us.get("key_event") or {}).get("state") or "")))
    sim = UserSimulator(
        us, llm_gateway=llm, tts_gateway=tts,
        llm_model="stub-judge-model", role="collection", dynamics="all_positive",
        leakage_label=leakage, user_gender="male", scenario_id=sid, seed=42, **kw)
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
        UserSimulator({"key_event": {"description": "x", "state": "cooperative"}},
                      MagicMock(), MagicMock())
    with pytest.raises(ValueError):
        UserSimulator(_script(key_event={"description": "x", "state": "N5_resistant"}),
                      MagicMock(), MagicMock())


def test_prompt_sections_and_stable_prefix():
    sim, _ = _sim(leakage="T1")  # T1 口径含关键事件参考词注入
    m1 = sim._build_messages(1, "", forced=False)
    m2 = sim._build_messages(1, "", forced=False)
    assert m1 == m2  # 稳定前缀逐字节可复现（缓存友好）
    user = m1[1]["content"]
    for section in ("【身份】", "【说话风格】", "【目标】", "【事实锚】",
                    "【用户本人掌握的事实】", "【披露时机】", "【剧情要点】",
                    "【关键事件】", "【透明度约束】", "【轮次预算】", "【完整对话历史】"):
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
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
           state="cooperative", is_key_turn=True, covered=[2, 3])
    r2 = sim.next_turn("您打算什么时候处理？")
    assert r2.is_key_turn and sim.key_completed
    _reply(llm, text="嗯，那渠道和到账时间你说一下吧。",
           state="cooperative", is_key_turn=True)
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
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
           state="cooperative", is_key_turn=True)
    sim.next_turn("您打算怎么处理？")
    _reply(llm, text="好的，那就这么说定了，麻烦你了，再见。", end_call=True)
    r = sim.next_turn("我帮您登记好了。")
    assert r.end_call and sim.termination_reason == "user_closed"
    assert sim.next_turn("再见") is None


def test_state_mismatch_overridden_with_data():
    sim, llm = _sim()
    # 关键轮 state 报错 → 重试仍报（mock 不变）→ 以数据为准覆盖并记 state_mismatch
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
           state="displeased", is_key_turn=True)
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


# ---- 4. TTS instruct 表驱动（Plan L §4.1 / §4.2）----

def test_instruct_lookup_verbatim_from_table():
    """查表命中 → instruct 与表内文本逐字一致（运行期不再拼任何韵律文本）。"""
    entry = fake_instruction_entry("cooperative")
    sim, llm = _sim(instruction_entry=entry)
    _reply(llm)
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.tts_control == entry["non_key"]["neutral"]
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
           state="cooperative", is_key_turn=True)
    r2 = sim.next_turn("您打算怎么处理？")
    assert r2.tts_control == entry["key"]  # 关键轮固定取 key 条（不分透明度）


def test_instruct_real_table_lookup():
    """真实表条目：base_scenario_id 由 scenario_id 剥尾缀推出，取值逐字来自表。"""
    table = load_table()
    base = "COL-ALL-01"
    entry = table.entry(base)
    sc = _script(key_event={"description": "摊牌", "state": entry["key_state"],
                            "trigger_hint": "x", "interrupt": False})
    sim, llm = _sim(user_script=sc, scenario_id=f"{base}-T2", instruction_entry=None)
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
    sc = _script(key_event={"description": "摊牌", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
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
    sim, _ = _sim(instruction_entry=fake_instruction_entry(
        "cooperative", states=["neutral", "cooperative"]))
    assert sim.candidate_states == ("neutral", "cooperative")
    with pytest.raises(ValueError, match="instruction 表缺 state"):
        sim._assemble_instruct("urgent", is_key=False)


def test_instruct_table_missing_file_raises_at_init(tmp_path):
    """表文件缺失 → 构造期就 raise（不等到第一轮合成才发现）。"""
    with pytest.raises(ValueError, match="instruction 表不存在"):
        UserSimulator(_script(), MagicMock(), MagicMock(), scenario_id="COL-ALL-01-T2",
                      instruction_table_path=tmp_path / "nope.json")


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
    """韵律三级拉平：同 base 同 state，T1/T2/T3 的 instruct 逐字一致。

    表的索引键不含 -T1/-T2/-T3，这条不变量由数据结构保证；本测试守住接线不跑偏。
    """
    non_key, key = [], []
    for level in ("T1", "T2", "T3"):
        sim, llm = _sim(leakage=level)
        _reply(llm)
        non_key.append(sim.next_turn("您好，是张三先生吗？").tts_control)
        _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
               state="cooperative", is_key_turn=True)
        key.append(sim.next_turn("您打算怎么处理？").tts_control)
    assert len(set(non_key)) == 1 and len(set(key)) == 1


# ---- 4b. 韵律因子臂 prosody_arm（Plan L §6 步骤 4 的 2×2 A/C 中性臂）----

def test_prosody_arm_neutral_is_state_and_keyturn_invariant():
    """中性臂：instruct 恒等于 non_key[neutral]，与本轮 state / is_key 全无关。

    这是 A/C 两格「韵律恒中性」的定义 —— 只有它能被主动产生，不再寄生于
    2026-08-26 那批 instruction 未下发的 bug。
    """
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"])
    sc = _script(key_event={"description": "拒绝安排", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
    sim, _ = _sim(user_script=sc, instruction_entry=entry, prosody_arm="neutral")
    neutral = entry["non_key"]["neutral"]
    assert sim._assemble_instruct("neutral", is_key=False) == neutral
    assert sim._assemble_instruct("displeased", is_key=False) == neutral  # 非中性 state 也拉平
    assert sim._assemble_instruct("displeased", is_key=True) == neutral   # 关键轮不取 key 条


def test_prosody_arm_neutral_drops_forced_tag():
    """中性臂：不附加任何强制标签（[angry]/[whispers] 本身即韵律，留着会污成半中性）。"""
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"],
                                   tags={"displeased": _DISPLEASED_TAG},
                                   key_tag=_DISPLEASED_TAG)
    sc = _script(key_event={"description": "拒绝安排", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
    sim, llm = _sim(user_script=sc, instruction_entry=entry, prosody_arm="neutral")
    assert sim._forced_tag("displeased", is_key=False) == ""
    assert sim._forced_tag("displeased", is_key=True) == ""
    _reply(llm, text="这个安排我接受不了，得再商量商量。", state="displeased", is_key_turn=True)
    r = sim.next_turn("您看这样安排行吗？")
    assert r.text_tts == r.text  # 无标签，两字段逐字一致
    assert r.text_tts == "这个安排我接受不了，得再商量商量。"


def test_prosody_arm_default_is_state():
    """默认臂 = state：不传 prosody_arm 时行为不变（关键轮仍取 key 条、带强制标签）。"""
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"],
                                   tags={"displeased": _DISPLEASED_TAG},
                                   key_tag=_DISPLEASED_TAG)
    sc = _script(key_event={"description": "拒绝安排", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
    sim, _ = _sim(user_script=sc, instruction_entry=entry)
    assert sim.prosody_arm == "state"
    assert sim._assemble_instruct("displeased", is_key=True) == entry["key"]
    assert sim._forced_tag("displeased", is_key=True) == _DISPLEASED_TAG


def test_prosody_arm_invalid_raises_at_init():
    """非法臂名 → 构造期 raise（只接 state / neutral）。"""
    with pytest.raises(ValueError, match="prosody_arm"):
        _sim(prosody_arm="middle")


# ---- 4c. 演用户 reasoning 开关 sim_thinking（planL 台账 C 第 7 项）----

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
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"],
                                   tags={"displeased": _DISPLEASED_TAG},
                                   key_tag=_DISPLEASED_TAG)
    sc = _script(key_event={"description": "拒绝安排", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
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
    entry = fake_instruction_entry("displeased", states=["neutral", "displeased"],
                                   tags={"displeased": _DISPLEASED_TAG},
                                   key_tag=_DISPLEASED_TAG)
    sc = _script(key_event={"description": "拒绝安排", "state": "displeased",
                            "trigger_hint": "x", "interrupt": False})
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
    _reply(llm, text=f"[{_DISPLEASED_TAG}][{_DISPLEASED_TAG}]我这边真的没法接受这个安排。",
           state="displeased")
    r = sim.next_turn("您看这样安排行吗？")
    assert r.text_tts.count(f"[{_DISPLEASED_TAG}]") == 1
    assert "[" not in r.text
    assert [x.split(":")[1] for x in r.llm_debug["tag_stripped_reason"]] == \
        ["not_at_head", "not_at_head"]


def test_runtime_no_emotion_template_reference():
    """运行期路径 0 引用 emotion.tts_control_for（模板拼装整体退役）。"""
    from pathlib import Path

    from listen2serve.runtime import user_simulator as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "tts_control_for" not in src
    for retired in ("MAX_TONE_CHARS", "RATE_SLOT", "VOLUME_SLOT", "_KEY_TURN_TONE_BY_STATE",
                    "_INTENSITY_NAME", "key_intensity"):
        assert retired not in src, retired


def test_opening_instruct_independent_of_level():
    """开场轮 instruct 不再随 leakage_label 变化（固定 neutral 查表）。"""
    controls = []
    for level in ("T1", "T2", "T3"):
        sim, _ = _sim(leakage=level, role_type="outbound_pressure",
                      scenario_id="COL-ALL-01")
        t1 = sim.next_turn("")
        controls.append(t1.tts_control)
        assert t1.state == "neutral"
    assert len(set(controls)) == 1


def test_state_drift_counted_not_overridden():
    """mission 声明的 state 与 LLM 自报不符 → 只记 state_drift，不覆盖（§4.4）。"""
    sc = _script(mission=[
        {"beat": "接听电话，确认自己是本人", "state": "neutral"},
        {"beat": "弄清来意与欠款情况", "state": "doubtful"},
        {"beat": "【关键事件】摊牌还款能力", "state": "cooperative", "key_event": True},
    ])
    sim, llm = _sim(user_script=sc)
    _reply(llm, state="neutral", covered=[2])  # mission 第 2 条要 doubtful
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.state == "neutral" and sim.stats["state_drift"] == 1
    assert r.llm_debug["state_expected"] == "doubtful"



# ---- 5. 词表防线 ----

def test_vocab_t2_zero_hit_defense():
    sim, llm = _sim(leakage="T2")
    # 首试命中情绪词 → 反馈重试；重试仍命中（mock 不变）→ 保留并标记
    _reply(llm, text="我很烦你们老是打电话过来催我还款。")
    r = sim.next_turn("您好，是张三先生吗？")
    assert r.vocab_violation and sim.stats["vocab_violation"] == 1
    assert sim.stats["retries"] == 1


def test_vocab_t1_key_positive_check():
    sim, llm = _sim(leakage="T1")
    # 关键轮未命中 cooperative 子表词 → 重试后标记（不阻断）
    _reply(llm, text="行，下月十号前我先还九千，其余宽限几天。",
           state="cooperative", is_key_turn=True)
    r = sim.next_turn("您打算怎么处理？")
    assert r.vocab_violation
    # 命中子表词 → 合规
    sim2, llm2 = _sim(leakage="T1")
    _reply(llm2, text="没问题，我按你说的来，下月十号前先还九千。",
           state="cooperative", is_key_turn=True)
    r2 = sim2.next_turn("您打算怎么处理？")
    assert not r2.vocab_violation


def test_vocab_t2_key_whitelist_floor():
    """K18c：T2 关键轮除零命中外还需命中白名单下限（validate_level_v2 的闲置下限）。"""
    # 不带任何缓和词的短硬结论句 → 零情绪词也判违规（T2 最典型的病灶）
    sim, llm = _sim(leakage="T2")
    _reply(llm, text="那就这样定下来，我下月十号前还九千。", state="cooperative",
           is_key_turn=True)
    r = sim.next_turn("您打算怎么处理？")
    assert r.vocab_violation
    # 带缓和句式 → 合规
    sim2, llm2 = _sim(leakage="T2")
    _reply(llm2, text="能不能先还九千，剩下的回头再商量？", state="cooperative",
           is_key_turn=True)
    assert not sim2.next_turn("您打算怎么处理？").vocab_violation
    # 下限只管关键轮：非关键轮仍只做零命中
    sim3, llm3 = _sim(leakage="T2")
    _reply(llm3, text="那就这样定下来，我下月十号前还九千。", state="cooperative")
    assert not sim3.next_turn("您打算怎么处理？").vocab_violation


# ---- 6. 打断预生成 ----

def test_peek_interrupt_draft_no_side_effect():
    sc = _script(key_event={"description": "摊牌", "state": "cooperative",
                            "trigger_hint": "x", "interrupt": True})
    sim, llm = _sim(user_script=sc)
    _reply(llm, text="你等等，我先说，下月十号前我先还九千行不行？",
           state="cooperative", is_key_turn=True, interrupt=True)
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


# ---- 7. Plan J §6：outbound 首轮被动接听 ----

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


# ---- 8. Plan J §5.1：v6.1 四块式 prompt schema ----

def test_v61_prompt_four_blocks():
    """四块式：你是谁/要办的事/怎么说话/输出协议；事实锚物理移除；硬约束 ≤4 条。"""
    sim, _ = _sim(prompt_schema="v6.1", background_consensus="客户在平台有欠款，双方商讨还款。")
    msgs = sim._build_messages(1, "", forced=False)
    user = msgs[1]["content"]
    for block in ("【你是谁】", "【要办的事】", "【怎么说话】", "【输出协议】"):
        assert block in user
    # 事实锚退役：不知情信息物理移除（不给即不知）
    assert "【事实锚】" not in user and "事实锚使用边界" not in user
    # 共识背景注入
    assert "双方商讨还款" in user
    # 硬约束 ≤4 条（①②③④）
    assert "④" in user and "⑤" not in user
    # 透明度约束与协议字段保留（词表防线不可破）
    assert "【透明度约束】" in user and "is_key_turn" in user


def test_v61_progress_and_budget():
    """动态尾部：进度/预算/历史/客服上轮与 v6.0 同构。"""
    sim, _ = _sim(prompt_schema="v6.1")
    user = sim._build_messages(2, "您好，请问有什么可以帮您？", forced=False)[1]["content"]
    assert "【进度】" in user and "【轮次预算】" in user
    assert "【客服上轮回复】您好，请问有什么可以帮您？" in user


# ---- 8b. Plan K §3.2：v7 叙事化（v6.1 四块式 + 三点增量）----

def test_v7_prompt_transparency_merged():
    """v7 增量①：透明度约束并入关键事件一句【语气提示】，不再单列【透明度约束】区块。"""
    sim, _ = _sim(prompt_schema="v7", background_consensus="客户在平台有欠款，双方商讨还款。")
    user = sim._build_messages(1, "", forced=False)[1]["content"]
    for block in ("【你是谁】", "【要办的事】", "【怎么说话】", "【输出协议】"):
        assert block in user
    assert "【透明度约束】" not in user  # 不再单列区块
    assert "【语气提示】" in user and "is_key_turn" in user  # 并入关键事件一句
    assert "双方商讨还款" in user  # 共识背景注入保留


def test_v7_history_compressed():
    """v7 增量②：历史压缩——近 8 条全文，更早的拼成【更早对话摘要】。"""
    from listen2serve.runtime.user_simulator import _render_history_compressed
    hist = ([{"role": "user" if i % 2 == 0 else "assistant",
              "content": f"第{i}轮内容" + "字" * 30} for i in range(12)])
    out = _render_history_compressed(hist)
    assert "【更早对话摘要】" in out
    # 近 8 条全文保留（第 4..11 轮）；更早的 4 条进摘要且被截断
    assert "第11轮内容" in out and "第4轮内容" in out
    assert out.count("字" * 30) == 8  # 仅近 8 条保留全文长度
    assert _render_history_compressed([]) == "（无）"
    sim, _ = _sim(prompt_schema="v7")
    sim._history = hist
    user = sim._build_messages(7, "好的", forced=False)[1]["content"]
    assert "【更早对话摘要】" in user and "【完整对话历史】" not in user


def test_v7_fewshot_in_how_block():
    """v7 增量③：few-shot 真实语料示范注入【怎么说话】（有则注入，无则优雅省略）。"""
    sim, _ = _sim(prompt_schema="v7")
    user = sim._build_messages(1, "", forced=False)[1]["content"]
    from listen2serve.runtime.user_simulator import _fewshot_examples
    if _fewshot_examples("collection", "all_positive"):
        assert "不学它们的态度和内容" in user  # Plan K 审阅：引导语由“不抄内容”加强
    assert "【怎么说话】" in user


def test_v7_t3_key_clause_and_fewshot_filter():
    """Plan K 纸面校验修正：T3 关键轮结论不上文本，且与硬约束不矛盾。"""
    from listen2serve.runtime.user_simulator import (
        _filter_examples_for_level, _transparency_key_clause_v7,
    )

    t3 = _transparency_key_clause_v7("T3", [])
    assert "结论一个字都不许说出口" in t3
    # 不再把“复述客服关键信息”当作中性载体写进子句（原因是当时的硬约束①一刀切
    # 禁复述；K18c 已按 planB 原意收窄该硬约束，但中性载体改由数据侧 surface 给定）
    assert "复述" not in t3
    # K18b 复跑修正（R1）：只禁结论会被过度矫正成反向提问，须附同向约束
    # K18c：同向约束改为立场无关表述（原句只兑得上负向关键事件）
    # planS1 改动4：同向约束从双向举例（“有兴趣…推脱或反悔”）压成一句原则，删掉例子
    # （改动5：常驻位置不放例句）。立场无关的同向约束本身逐字保留，故断言改为匹配新措辞。
    assert "顺着你在【关键事件】里的立场" in t3
    assert "不能说成反方向" in t3
    # T3 过滤结论式示范句；余下不足 2 条则保留原样
    raw = ["我不需要，以后不要再打来了。", "嗯，那你把办理方式说一下。", "这个活动什么时候到期？"]
    kept = _filter_examples_for_level(raw, "T3")
    assert kept == raw[1:]
    assert _filter_examples_for_level(raw, "T1") == raw
    assert _filter_examples_for_level(["我不需要", "别再打了"], "T3") == ["我不需要", "别再打了"]


def test_v7_key_event_marked_not_script():
    """Plan K 审阅④：关键事件描述显式标为意图说明，防模型直接照念结论。"""
    sim, _ = _sim(prompt_schema="v7")
    user = sim._build_messages(1, "", forced=False)[1]["content"]
    assert "不是台词，别照着念" in user


def test_v7_key_completed_no_restate_hint():
    """K18b 复跑修正（R2）：关键事件完成后禁回头补说结论，key_downgrade 收敛。"""
    sim, _ = _sim(prompt_schema="v7")
    before = sim._build_messages(2, "您看具体哪天能还呢？", forced=False)[1]["content"]
    assert "【关键事件已完成】" not in before
    sim._key_turn_no = 2  # key_completed 为只读属性（由关键轮号推导）
    assert sim.key_completed
    after = sim._build_messages(3, "您看具体哪天能还呢？", forced=False)[1]["content"]
    assert "【关键事件已完成】" in after
    assert "is_key_turn 一律填 false" in after


def test_v7_system_role_guard_intact():
    """v7 system 提示词的双方隔离约束不得被误改（曾因编辑引入错别字）。"""
    sim, _ = _sim(prompt_schema="v7")
    system = sim._build_messages(1, "", forced=False)[0]["content"]
    assert "绝不能说客服的台词" in system


def test_v7_key_clause_reads_surface_all_levels():
    """K18c：三层关键轮子句接入数据侧 surface，缺字段时回落原通用话术。"""
    from listen2serve.runtime.user_simulator import _transparency_key_clause_v7

    sfc = "只问缴费方式和到账期限这两件事的具体说法，不表态"
    for level in ("T1", "T2", "T3"):
        with_sfc = _transparency_key_clause_v7(level, ["恶心"], sfc)
        assert with_sfc.startswith(sfc + "。"), level
        # 渲染拼接后不得出现嵌套冒号或语义重复（纸面校验查出过两处）
        assert "照这个说法办：" not in with_sfc, level
        assert with_sfc.count("把这份情绪说出来") == 0, level
    # 缺 surface 时回落：未改写的场景照旧可跑
    assert "把这份情绪说出来" in _transparency_key_clause_v7("T1", ["恶心"])
    assert "中性的业务句子" in _transparency_key_clause_v7("T3", [])


def test_v7_t2_key_clause_has_positive_floor():
    """K18c：T2 不再是“T3 减去结论禁令”，须有正向下限使三层泄露量单调。"""
    from listen2serve.runtime.user_simulator import _transparency_key_clause_v7

    t2 = _transparency_key_clause_v7("T2", [])
    assert "必须带上至少一个这类词或缓和句式" in t2  # 下限（该说什么）
    assert "不把态度结论说全" in t2  # 泄露量卡在 T1（说全）与 T3（不说）之间


def test_v7_hard_constraint_allows_spoken_restate():
    """K18c：硬约束①按 planB 原意收窄到“禁原样照搬”，不再一刀切禁复述。"""
    sim, _ = _sim(prompt_schema="v7")
    user = sim._build_messages(1, "", forced=False)[1]["content"]
    assert "不得原样照搬客服的话" in user
    assert "可以用自己的口语把关键信息复述确认" in user
    assert "不得复读或改写客服刚说的话" not in user


def test_v7_pre_key_stance_ban_asymmetry_fixed():
    """K18b 复跑修正（R3）：非关键轮也有提前摊牌禁令，堵 P4；T1 不加（asymmetry）。

    planS1 改动4/5 更新（匹配新措辞、不削弱意图）：禁令从「枚举禁用句式清单」压成
    「一句原则」——改动5 明确常驻位置不放词表（planN 实测例词清单被当成逐轮推荐模板，
    填充词起头率 24%→92%）。asymmetry（T1 空、T2/T3 非空）与「提前摊牌延后到关键轮」的
    意图逐字不变；枚举词表防线移到代码侧 _T3_STANCE_MARKERS（§5.3 不动 tuple），故不再
    断言 prompt 里出现「不打算」等具体词——改断言原则句在位 + asymmetry 在位。
    """
    from listen2serve.runtime.user_simulator import _pre_key_stance_ban_v7

    assert _pre_key_stance_ban_v7("T1") == ""  # T1 本就要显式态度，不加禁令（asymmetry 一半）
    for level in ("T2", "T3"):
        ban = _pre_key_stance_ban_v7(level)
        assert ban != ""                 # 非关键轮禁令必须在位（asymmetry 另一半）
        assert "那轮之前先按住" in ban     # 提前摊牌延后到关键轮的原则句（堵 P4 的意图不变）
