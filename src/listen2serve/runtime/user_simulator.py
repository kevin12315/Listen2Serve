"""骨架驱动的自由演绎用户模拟器（内部方案 v6.0 数据契约）。

输入：scenario.user_script（v6.0 骨架：mission 要点清单 + key_event 关键事件 +
disclosure_plan 披露时机 + turn_budget 软/硬预算）+ Agent 上轮回复。

设计原则与客服侧对齐：强 LLM + 语义级约束 + 全上下文。

每轮一次 LLM 调用，输出紧凑 JSON 协议（text/state/is_key_turn/
end_call/covered/interrupt），运行时硬校验 + 确定性状态机：
- 唯一关键轮保障：LLM 自报 is_key_turn，首次 true 锁存；重复 true 降级；
  soft 轮未触发升级 prompt；hard-2 轮未触发强制置位（forced_key_turn）；
- 透明度分级：非关键轮统一「克制中性」，T1/T2/T3 差异集中在关键轮
  （词表正向要求仅 T1 关键轮；T2/T3 全轮 + T1 非关键轮零命中防线）；
- TTS instruct：内部方案 起改为按 base_scenario_id × state 查预生成表
  （data/benchmark/tts_instructions.json），运行期不拼任何韵律文本、查不到即硬失败；
  情感标签由表强制附加并过四重校验（text 落盘、text_tts 送合成）；
- 轮次：不预定轮次，soft 进双侧 prompt、hard 进 orchestrator max_turns；
  end_call 自然收尾（须在关键事件完成后合法），termination_reason=user_closed；
- 打断：key_event.interrupt=true 且关键事件未完成时，peek_interrupt 预生成
  下一轮草案取其 interrupt 标记（编排器打断窗口逻辑沿用）。

Token 经济性（）：prompt 拆稳定前缀（system+身份/风格/目标/事实/披露/
mission/关键事件定义/透明度，整场逐字节不变）+ 动态尾部（进度/预算/历史）；
非 reasoning 后端 max_tokens 2048→320（finish_reason=length 视为协议违规，
以 1024 上限重试一次），Gemini reasoning 保留 2048；协议违规合并为一次反馈重试。
"""

from __future__ import annotations

import json
import logging
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listen2serve.audio.effects import EffectConfig, apply_effects
from listen2serve.domains.base import STATE_LABELS, normalize_state
from listen2serve.evaluation.closing_check import _USER_CLOSING
from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.gateway.tts_gateway import TTSGateway
from listen2serve.runtime.tts_instructions import load_table as load_instruction_table
from listen2serve.runtime.tts_instructions import validate_entry as validate_instruction_entry
from listen2serve.runtime.tts_tags import allowed_tags_from_entry, apply_tag_policy, find_tags

logger = logging.getLogger(__name__)

MIN_UTTERANCE_CHARS = 10 # 统一字数下限（含标点；不再分关键/非关键轮口径）
MAX_UTTERANCE_CHARS = 60 # 统一字数上限（含标点）
MAX_TTS_CONTROL_CHARS = 80 # tts_control 长度硬上限（「用{…}的语气」包装契约沿用）

# 括号舞台指示（2026-09-06 新增）。演绎模型有时把 prompt 里的舞台指示回吐进 text，
# 形如「（压低声音）嗯……本期物业费是八千六百元。」。实测归因（三重证据）：
# qwen-plus 产的 275 条 canonical 台词 0 条含括号、其 46 通历史多轮 run 0 通含括号，
# 而换 某带长思考的推理模型 后的 15 通里有 1 通含括号且该通 8 个用户轮中 6 轮都带 ⇒ 是模型行为差异。
# 危害分三层，必须分开看：
# ① 音频刺激**不受影响** —— TTS（qwen-audio-3.0-tts-plus）自己把括号当舞台指示吞掉，
# ASR 转写落盘 wav 实证转写结果里没有「压低声音」；
# ② **所有读文本的下游被污染** —— 纯文本裁判（裁定 8：wav=None）、泄漏探针、人工审、
# 论文引台词，拿到的都是带括号的原文；
# ③ 已实际污染一个读数：pilotW15 那 15 通里「仅凭文字猜中真实状态 1/15」的**唯一**命中，
# 正好就是**唯一**带括号的那条（COL-ALL-56-T3，真值 hushed）⇒ 剔除后真值 0/14。
# 即被测输入里混进了答案，是本仓登记过的「needle 自污染」同一类事故。
# 处置选择在**代码侧剥离**而不是「prompt 里加禁令」：内部方案 已实测枚举禁令会被演绎模型
# 当成推荐模板（填充词起头率 24%→92%），与 内部方案 那条裁定的逻辑一致。
# ⚠️ 剥掉的数量必须记进 stats（stage_direction_stripped），**不许静默剥**；
# 剥完正文为空则落入下面既有的「text 为空」分支 ⇒ 判为一次演绎失败并重试，不落空台词。
STAGE_DIRECTION_RE = re.compile(r"[（(][^（）()]*[)）]")

# 6 态枚举（domains.base.STATE_LABELS 同码）。内部方案：运行期的 state 值域收窄为
# 本 base 的 candidate_states（进 prompt 与硬校验），本元组只做全局合法性兵库。
STATES = ("neutral", "cooperative", "displeased", "urgent", "doubtful", "hushed")

# ---- 内部方案：outbound 开场修正（催收/营销外呼：用户首轮被动接听） ----
# 应答池据 真实通话开场聚类（喂/喂你好/哎你好 绝对主导）；
# crc32(scenario_id) 确定性选取，不注入 mission（mission 第 2 轮起生效）。
OUTBOUND_ROLE_TYPES = ("outbound_pressure", "outbound_negotiation")
OUTBOUND_OPENING_POOL = (
    "喂，你好。", "喂？", "喂，你好？", "哎，你好。", "你好。",
    "喂，哪位？", "你好，哪位？", "嗯，你好。",
)

# 韵律关键词（与 _prosody_from_control 提取对齐；也是 _instruct_from_control 的契约判据）。
# 内部方案：原先「LLM 自填 tone/rate/volume 槽位 + 代码按 state 查模板拼装 + 强度
# 分档命名」那一整套已整体退役（连同 emotion 模块里对应的常量与拼装函数）：instruct
# 改为按 base_scenario_id × state 查预生成表，运行期不再拼任何韵律文本。
RATE_WORDS = ("语速中等", "语速偏快", "语速快", "语速很快")
VOLUME_WORDS = ("音量正常", "音量略升", "音量大", "音量压低")


@dataclass
class UserTurnResult:
    """模拟器一轮输出（v6.0 协议字段）。"""

    text: str = ""
    # 内部方案：实际送 TTS 的文本（可能带句首标签）；text 为剥标签后的干净台词，
    # 评委打分/ASR 对齐一律用 text，二者分离避免标签字面进入文本类指标。
    text_tts: str = ""
    pcm16: bytes = b""
    state: str = ""
    tts_control: str = ""
    is_key_turn: bool = False
    # 内部方案：本轮关键轮句式（question/statement），仅关键轮非空。B−C 的分析必须能按
    # 句式分层 —— 问句本身携带 doubtful 先验，不分层则韵律效应与句式效应混在一起。
    key_form: str = ""
    covered: list[int] = field(default_factory=list)
    forced_key_turn: bool = False
    end_call: bool = False
    interrupt: bool = False # 本轮以打断方式开口（仅关键轮合法）
    state_mismatch: bool = False # 关键轮 state 与数据不符被覆盖
    vocab_violation: bool = False # 词表防线重试后仍违规（不阻断，仅标记）
    llm_debug: dict[str, Any] = field(default_factory=dict)


class UserSimulator:
    """骨架驱动的自由演绎用户模拟器（骨架确定剧情与关键事件，LLM 自由演绎全程话术）。"""

    def __init__(
        self,
        user_script: dict[str, Any],
        llm_gateway: LLMGateway | None = None,
        tts_gateway: TTSGateway | None = None,
        llm_model: str | None = None,
        voice: str | None = None,
        audio_mode: str = "control",
        seed: int = 0,
        effect_config: EffectConfig | None = None,
        role: str = "",
        dynamics: str = "",
        db_seed: dict[str, Any] | None = None,
        scenario_id: str = "",
        leakage_label: str = "",
        user_gender: str = "",
        sub_domain: str = "",
        prompt_log_path: str | Path | None = None,
        role_type: str = "",
        prompt_schema: str = "v6.0",
        background_consensus: str = "",
        base_scenario_id: str = "",
        instruction_entry: dict[str, Any] | None = None,
        instruction_table_path: str | Path | None = None,
        prosody_arm: str = "state",
        sim_thinking: bool = False,
    ) -> None:
        self.user_script = user_script
        self.llm = llm_gateway or LLMGateway()
        self.tts = tts_gateway or TTSGateway()
        self.llm_model = llm_model
        self.voice = voice
        self.audio_mode = audio_mode
        self.seed = seed
        self.effect_config = effect_config
        self.role = role
        self.dynamics = dynamics or user_script.get("dynamics", "")
        self.db_seed = db_seed
        self.scenario_id = scenario_id
        self.leakage_label = normalize_leakage_level(leakage_label)
        self.user_gender = user_gender
        self.sub_domain = sub_domain
        # 内部方案：逐轮 elicit prompt 落盘（独立 jsonl，不塞 sim_status 避免膨胀）
        self.prompt_log_path = Path(prompt_log_path) if prompt_log_path else None
        self.role_type = role_type
        # 内部方案：演绎 prompt schema（v6.0 旧十一区块 / v6.1 四块式减负；A/B 达标后切默认）
        self.prompt_schema = prompt_schema
        self.background_consensus = background_consensus
        # 内部方案 步骤 4：2×2 四格的韵律因子臂。"state" = 带 state 韵律（B / D 两格，
        # 即正式口径）；"neutral" = 韵律中性（A / C 两格）。这不是“新旧双口径开关”（决策 c 禁止
        # 的那个）：旧口径是“韵律由演绎 LLM 现填”，本开关两臂都是表驱动，只差在查哪一条。
        # 它必须存在：四格设计需要中性臂能被**主动产生**，而 2026-08-26 那批 run 的中性是
        # instruction 未下发这个 bug 顺带造成的 —— 把对照臂寄在 bug 上不可重现。
        if prosody_arm not in ("state", "neutral"):
            raise ValueError(f"prosody_arm 非法: {prosody_arm!r}（只接 state / neutral）")
        self.prosody_arm = prosody_arm

        # planL 台账 C 第 7 项：只作用于**演用户的** LLM。被测 agent 走 realtime 音频通路，
        # 不吃这个参数，所以这一格测的是「用户演得更自洽吗」，不是「客服答得更好吗」。
        self.sim_thinking = sim_thinking

        # ---- v6.0 骨架构件 ----
        raw_mission = user_script.get("mission") or []
        # 内部方案：mission 升级为 list[{beat,state?,key_event?}]。两种格式并存期先归一
        # （纯字符串条目视为 {"beat": <str>}），下游只看 beat 文本，渲染结果逐字不变。
        self.mission_entries: list[dict[str, Any]] = [
            dict(m) if isinstance(m, dict) else {"beat": str(m)} for m in raw_mission
        ]
        self.mission: list[str] = [str(e.get("beat") or "") for e in self.mission_entries]
        self.key_event: dict[str, Any] = user_script.get("key_event") or {}
        self.disclosure_plan: dict[str, Any] = user_script.get("disclosure_plan") or {}
        budget = user_script.get("turn_budget") or {}
        self.soft_budget = int(budget.get("soft") or 6)
        self.hard_budget = int(budget.get("hard") or 12)
        if not self.mission or not self.key_event.get("description"):
            raise ValueError("user_script 缺 mission/key_event（v6.0 数据契约）")
        key_state = str(self.key_event.get("state") or "")
        if key_state not in STATES:
            raise ValueError(f"key_event.state 非法: {key_state!r}")

        # ---- 内部方案：TTS instruction 表（表驱动韵律，构造期就把表查好）----
        # 解析顺序：显式注入条目 > 按 base 查表 > raise。没有静默兜底（）：回落到
        # 模板拼装正是上一版把「文本平静、声音愤怒」的错配藏了一整轮的原因。
        # instruction_entry 是给合成场景（单测/导出 demo）的注入口：它们的 scenario_id
        # 不在表里，但也不应因此获得「查不到就拼个」的特权。
        self.base_scenario_id = base_scenario_id or _base_scenario_id_of(scenario_id)
        entry = dict(instruction_entry) if instruction_entry is not None else \
            load_instruction_table(instruction_table_path).entry(self.base_scenario_id)
        validate_instruction_entry(self.base_scenario_id, entry)
        if entry["key_state"] != key_state:
            raise ValueError(
                f"instruction 表 key_state={entry['key_state']!r} 与 key_event.state={key_state!r} "
                f"不一致：base={self.base_scenario_id}"
            )
        self._instruction_entry = entry
        # state 值域收窄到本 base 的候选集（进 prompt 与硬校验）
        self.candidate_states: tuple[str, ...] = tuple(entry["candidate_states"])
        # 表里声明的强制标签集（当前数据下只有 14 个 base 非空）：不进 prompt，由代码前置，
        # 但必须进四重校验的允许集，否则会被自己的规则 1 剔掉。
        self._allowed_tags = allowed_tags_from_entry(entry)
        # T1 关键轮正向校验抽样词（确定性：seed 步进取 10 词，进稳定前缀 prompt）
        self._t1_sample_words = self._sample_state_words(key_state)

        # ---- 确定性状态机 ----
        self._turn_count = 0
        self._key_turn_no: int | None = None # 锁存的唯一关键轮号
        self._covered: set[int] = set()
        self._ended = False
        self.termination_reason = "" # user_closed |（硬上限由 orchestrator 标 max_turns）
        self._history: list[dict[str, str]] = []
        self._last_state = "neutral" # 上一轮实际 state（_state_for_turn 算预期值用）
        # 打断预生成草案（peek_interrupt 用；next_turn 时丢弃重生成）
        self._draft_done = False
        self._draft_interrupt = False
        # 协议健康度计数（reflect_smoke 消费）
        self.stats = {
            "llm_calls": 0, "json_failures": 0, "retries": 0, "truncation_retries": 0,
            "forced_key_turn": 0, "state_mismatch": 0, "vocab_violation": 0,
            "end_call_rejected": 0, "key_downgrade": 0, "outbound_opening": 0,
            # planP：end_call=true 但台词里没把收尾说出口的轮次。它记的是演绎侧保真度
            # （用户有没有让客服听得出要挂），与 closing_dangling 记的客服合规性是两件事。
            "end_call_no_cue": 0,
            # 内部方案：state_drift = LLM 声明值与 mission 预期值不符（只记不覆盖）；
            # tag_stripped = 四重校验剔掉的标签数（LLM 自行输出标签即违约）。
            "state_drift": 0, "tag_stripped": 0,
            # 2026-09-06：剥掉的括号舞台指示个数（见 STAGE_DIRECTION_RE）。
            # 与 tag_stripped 同理 —— 都是「模型输出了不该有的东西、被代码侧清掉」的计数，
            # 必须可见，否则换演绎模型后这类行为漂移没人发现。
            "stage_direction_stripped": 0,
        }

    # ---- 骨架构件辅助 ----
    def _sample_state_words(self, state: str) -> list[str]:
        """key_event.state 子表确定性抽样 8–12 词（T1 关键轮正向校验与 prompt 注入）。"""
        from listen2serve.vocab import state_pool

        pool = state_pool(state) or []
        if not pool:
            return []
        n = min(10, len(pool))
        seen: list[str] = []
        for i in range(len(pool)):
            w = pool[(self.seed + i * 7) % len(pool)]
            if w not in seen:
                seen.append(w)
            if len(seen) >= n:
                break
        return seen

    @property
    def key_form(self) -> str:
        """本场景关键轮句式（question / statement），内部方案 同态内配平。"""
        return key_form_for(self.scenario_id)

    @property
    def key_completed(self) -> bool:
        return self._key_turn_no is not None

    def _user_role_phrase(self) -> str:
        """与生成期同款角色短语（「一位{年龄}岁{性别}{身份词}」，≤20 字）。"""
        gender = {"male": "男性", "female": "女性"}.get(self.user_gender, "")
        m = re.search(r"(\d+)\s*岁", self.user_script.get("identity", ""))
        age = f"{m.group(1)}岁" if m else ""
        if self.role == "collection" and self.sub_domain == "物业费催缴":
            word = "业主客户"
        else:
            word = _ROLE_IDENTITY_WORD.get(self.role, "普通电话用户")
        return f"一位{age}{gender}{word}"[:20]

    # ---- 主流程 ----
    def next_turn(self, agent_reply: str = "", tool_calls: list[dict[str, Any]] | None = None) -> UserTurnResult | None:
        """生成下一轮用户发言（文本 + 音频）；end_call 轮之后返回 None。"""
        if self._ended:
            return None
        self._draft_done = False # 新客服轮：打断草案缓存失效
        n = self._turn_count + 1
        # 内部方案：outbound 首轮固定被动接听应答（不注入 mission，mission 第 2 轮起生效）
        if n == 1 and not agent_reply and self.role_type in OUTBOUND_ROLE_TYPES:
            return self._outbound_opening_turn(n)
        forced = (not self.key_completed) and n >= self.hard_budget - 2

        parsed = self._elicit(n, agent_reply, forced=forced)

        text = parsed["text"]
        state = self._state_for_turn(parsed)
        is_key = parsed["is_key_turn"]
        end_call = parsed["end_call"]
        covered_new = parsed["covered"]

        # ---- TTS instruct 查表（：韵律的唯一输入是 base_scenario_id × state）----
        tts_control = self._assemble_instruct(state, is_key)
        tag_raw = find_tags(text) # LLM 自行输出的标记（协议已删该字段，属违约）先留档
        forced_tag, tag_res = self._tagged_text(text, state, is_key)
        text = tag_res.text
        pcm16 = self._synthesize(tag_res.text_tts, state, tts_control)

        # ---- 状态机落账 ----
        if is_key:
            self._key_turn_no = n
        self._covered.update(covered_new)
        self._turn_count = n
        self._last_state = state
        if end_call:
            self._ended = True
            self.termination_reason = "user_closed"
            # 只计数：驳回 end_call 会改变轮数分布（轮数本身是上报指标），
            # 改写台词会破坏口语自然度。先看 prompt 约束够不够。
            if not any(w in text for w in _USER_CLOSING):
                self.stats["end_call_no_cue"] += 1

        if agent_reply:
            self._history.append({"role": "assistant", "content": agent_reply})
        self._history.append({"role": "user", "content": text})

        return UserTurnResult(
            text=text, text_tts=tag_res.text_tts, pcm16=pcm16, state=state,
            tts_control=tts_control,
            is_key_turn=is_key, covered=covered_new,
            key_form=self.key_form if is_key else "",
            forced_key_turn=parsed.get("_forced", False),
            end_call=end_call, interrupt=parsed.get("interrupt", False),
            state_mismatch=parsed.get("_state_mismatch", False),
            vocab_violation=parsed.get("_vocab_violation", False),
            llm_debug={
                "state_expected": parsed.get("_state_expected", ""),
                "tag_forced": forced_tag,
                "tag_raw": tag_raw,
                "tag_applied": tag_res.applied or "",
                "tag_stripped_reason": [f"{t}:{r}" for t, r in tag_res.stripped],
            },
        )

    # ---- 内部方案：outbound 首轮被动接听（消除「没人问就自答」开场伪影）----
    def _outbound_opening_turn(self, n: int) -> UserTurnResult:
        """外呼场景用户首轮：从应答池 crc32 确定性选一句，不走 LLM/不注入 mission。

        应答池据 真实通话统计（1069 通）：开场几乎全为「喂/喂，你好」类
        被动接听；架构不动（用户「喂」即首轮音频，WSS 触发链路不变）。
        """
        idx = zlib.crc32(self.scenario_id.encode("utf-8")) % len(OUTBOUND_OPENING_POOL)
        text = OUTBOUND_OPENING_POOL[idx]
        # 开场轮固定 neutral：表里 neutral 是 144/144 全覆盖的候选态，且开场句与
        # 透明度无关——原先按 leakage_label 取模板会让同一 base 的开场音色随 T1/T2/T3
        # 漂移，正是三档不可比的一个来源（）。
        state = "neutral"
        tts_control = self._assemble_instruct(state, is_key=False)
        # TTS 后端对个别 音色×超短文本 组合存在确定性缺陷（如「喂？」×longfengmeihui
        # 触发 cosyvoice InternalError）：合成失败时确定性顺延池内下一句重试，
        # 全池耗尽再抛错（落盘文本以实际合成句为准）。
        tts_err: Exception | None = None
        for k in range(len(OUTBOUND_OPENING_POOL)):
            cand = OUTBOUND_OPENING_POOL[(idx + k) % len(OUTBOUND_OPENING_POOL)]
            try:
                pcm16 = self._synthesize(cand, state, tts_control)
                text = cand
                tts_err = None
                break
            except Exception as exc: # noqa: BLE001 - 换句重试；耗尽后抛原错
                tts_err = exc
                logger.warning("开场 TTS 失败换句重试: scenario=%s text=%r err=%s",
                               self.scenario_id, cand, str(exc)[:80])
        if tts_err is not None:
            raise RuntimeError(
                f"outbound 开场 TTS 全池失败: scenario={self.scenario_id} last={tts_err}")
        self._turn_count = n
        self.stats["outbound_opening"] += 1
        self._history.append({"role": "user", "content": text})
        # 溯源对齐：开场轮也落一行（无 messages，标记 outbound_opening）
        if self.prompt_log_path is not None:
            try:
                record = {"turn": n, "outbound_opening": True, "forced": False,
                          "retried": False, "llm_params": None,
                          "messages": [{"role": "user", "content": text}]}
                self.prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.prompt_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                logger.warning("user_prompts 落盘失败（降级跳过）：scenario=%s err=%s",
                               self.scenario_id, exc)
        return UserTurnResult(
            text=text, text_tts=text, pcm16=pcm16, state=state, tts_control=tts_control,
            is_key_turn=False, covered=[], forced_key_turn=False,
            end_call=False, interrupt=False, state_mismatch=False,
            vocab_violation=False, llm_debug={"outbound_opening": True},
        )

    # ---- 打断预生成（orchestrator 在 Agent 说话中途调用，无状态副作用）----
    def peek_interrupt(self, agent_reply: str = "", tool_calls: list[dict[str, Any]] | None = None) -> bool:
        """关键事件未完成且 key_event.interrupt=true 时，预生成下一轮草案取
        interrupt 标记；每客服轮至多预生成一次（部分转写 ≥8 字后）。"""
        if (not self.key_event.get("interrupt") or self.key_completed
                or self._ended):
            return False
        if self._draft_done:
            return self._draft_interrupt
        if len(agent_reply or "") < 8:
            return False
        n = self._turn_count + 1
        forced = n >= self.hard_budget - 2
        try:
            draft = self._elicit(n, agent_reply, forced=forced, dry_run=True)
            self._draft_interrupt = bool(draft.get("interrupt", False))
        except Exception as exc: # noqa: BLE001 - 预生成失败不打断（保守）
            logger.warning("interrupt 预生成失败: %s", exc)
            self._draft_interrupt = False
        self._draft_done = True
        return self._draft_interrupt

    # ---- 每轮 LLM 协议生成 + 校验（合并为一次反馈重试）----
    def _elicit(self, n: int, agent_reply: str, *, forced: bool, dry_run: bool = False) -> dict[str, Any]:
        """一轮协议生成：首次调用 → 汇总硬违规 → 带反馈重试一次 → 残余按策略处置。

        硬失败（两次后仍 JSON 不可解析/空文本/复读客服/字数越界/state 非法）→
        抛错，由批跑层任务级重试（评测通路无兜底原则不变）。
        """
        messages = self._build_messages(n, agent_reply, forced=forced)
        parsed, problems = self._call_and_validate(messages, n, agent_reply, seed_offset=0)
        # 词表防线首轮命中也并入同一次反馈重试（：合并重试不串行）
        if parsed is not None:
            first_is_key = bool(parsed.get("is_key_turn")) and not self.key_completed
            first_hits = self._vocab_defense(parsed["text"], forced or first_is_key)
            if first_hits:
                problems.append(
                    f"台词命中词表防线（{first_hits}）：本轮禁止任何明确情绪词"
                    "（T1 关键轮须改用要求状态的明确情绪词），请改写后重新输出")
        if problems and not dry_run:
            self.stats["retries"] += 1
        if problems:
            retry_note = (
                "你刚才的输出不合格：" + "；".join(problems)
                + "。请重新输出合规的紧凑 JSON。"
            )
            retry_messages = messages + [
                {"role": "assistant", "content": "(上次输出不合格)"},
                {"role": "user", "content": retry_note},
            ]
            parsed2, _problems2 = self._call_and_validate(
                retry_messages, n, agent_reply, seed_offset=500)
            if parsed2 is not None:
                parsed = parsed2
            if parsed is None:
                raise RuntimeError(
                    f"用户模拟器协议两次失败：scenario={self.scenario_id} turn={n}")
            hard_left = self._hard_violations(parsed, n, agent_reply)
            if hard_left:
                raise RuntimeError(
                    f"用户模拟器协议重试后仍硬违规：scenario={self.scenario_id} "
                    f"turn={n} problems={hard_left}")

        if parsed is None:
            raise RuntimeError(
                f"用户模拟器协议两次失败：scenario={self.scenario_id} turn={n}")

        # ---- 唯一关键轮状态机（）----
        is_key = bool(parsed.get("is_key_turn"))
        if forced:
            if not is_key:
                parsed["_forced"] = True
                if not dry_run:
                    self.stats["forced_key_turn"] += 1
            is_key = True
            parsed["is_key_turn"] = True
            parsed["state"] = str(self.key_event.get("state"))
        if is_key and self.key_completed:
            # 重复声明 → 运行时降级为 false（唯一性硬保证）
            is_key = False
            parsed["is_key_turn"] = False
            parsed.pop("_forced", None)
            if not dry_run:
                self.stats["key_downgrade"] += 1
            logger.warning("is_key_turn 重复声明被降级：scenario=%s turn=%s", self.scenario_id, n)

        # ---- 关键轮 state 一致性（重试仍不等 → 以数据为准覆盖并记录）----
        if is_key and parsed["state"] != self.key_event.get("state"):
            parsed["_state_mismatch"] = True
            parsed["state"] = str(self.key_event.get("state"))
            if not dry_run:
                self.stats["state_mismatch"] += 1

        # ---- end_call 合法性（须在关键事件完成之后）----
        end_call = bool(parsed.get("end_call"))
        key_done_after = self.key_completed or is_key
        if end_call and not key_done_after:
            parsed["end_call"] = False
            parsed["_end_call_rejected"] = True
            end_call = False
            if not dry_run:
                self.stats["end_call_rejected"] += 1

        # ---- interrupt 合法性（仅关键轮 ∧ key_event.interrupt）----
        parsed["interrupt"] = bool(
            parsed.get("interrupt") and is_key and self.key_event.get("interrupt"))

        # ---- 词表运行时防线终判（）----
        hits = self._vocab_defense(parsed["text"], is_key)
        if hits is not None:
            parsed["_vocab_violation"] = True
            if not dry_run:
                self.stats["vocab_violation"] += 1
            logger.warning(
                "vocab violation（重试后仍命中，保留输出）：scenario=%s turn=%s hits=%s",
                self.scenario_id, n, hits)
        # ---- 内部方案：逐轮 elicit 溯源落盘（审计/调试用，不影响评测链路）----
        if not dry_run and self.prompt_log_path is not None:
            self._log_elicit(n, messages, retried=bool(problems), forced=forced)
        return parsed

    def _log_elicit(self, n: int, messages: list[dict[str, str]], *,
                    retried: bool, forced: bool) -> None:
        """逐轮一行追加 user_prompts.jsonl（turn/messages/llm_params/重试记录）；
        写失败不阻断主流程（溯源产物降级丢失，记 warning）。"""
        try:
            model_name = (self.llm_model or "").lower().split("/")[-1]
            record = {
                "turn": n,
                "forced": forced,
                "retried": retried,
                "llm_params": {
                    "model": self.llm_model,
                    "temperature": 0.7,
                    "max_tokens": 2048 if model_name.startswith("gemini") else 320,
                    "seed": self.seed * 1000 + n,
                    "thinking": self.sim_thinking,
                },
                "messages": messages,
            }
            self.prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.prompt_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno()) # NAS 写入时序防护（与 PartialStore 同口径）
        except OSError as exc:
            logger.warning("user_prompts 落盘失败（降级跳过）：scenario=%s turn=%s err=%s",
                           self.scenario_id, n, exc)

    def _hard_violations(self, parsed: dict[str, Any], n: int, agent_reply: str) -> list[str]:
        """硬字段残余违规清单（字数/复读/state 枚举）。"""
        problems: list[str] = []
        text = parsed.get("text") or ""
        if not (MIN_UTTERANCE_CHARS <= len(text) <= MAX_UTTERANCE_CHARS):
            problems.append(f"字数 {len(text)} 越界")
        if agent_reply and text and (text in agent_reply or agent_reply[:15] in text):
            problems.append("与客服上轮回复高度重合")
        if parsed.get("state") not in self.candidate_states:
            problems.append(
                f"state 非法: {parsed.get('state')!r}（本场景候选集 "
                f"{'/'.join(self.candidate_states)}）")
        return problems

    def _call_and_validate(
        self, messages: list[dict[str, str]], n: int, agent_reply: str, seed_offset: int,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """一次 LLM 调用：JSON 解析 + 硬字段校验。返回 (parsed 或 None, 违规清单)。

        max_tokens 分档（）：非 reasoning 后端 320（≈2 倍余量），
        finish_reason=length 视为协议违规以 1024 上限重试一次；
        Gemini reasoning 保留 2048。
        """
        model_name = (self.llm_model or "").lower().split("/")[-1]
        reasoning = model_name.startswith("gemini")
        limit = 2048 if reasoning else 320
        result = self.llm.chat(
            messages, model=self.llm_model, temperature=0.7, max_tokens=limit,
            seed=self.seed * 1000 + n + seed_offset,
            #：协议 JSON 输出默认不开 thinking（成本红线）。planL 台账 C 第 7 项开它做对照。
            thinking=self.sim_thinking,
        )
        self.stats["llm_calls"] += 1
        if result.finish_reason == "length" and not reasoning:
            self.stats["truncation_retries"] += 1
            result = self.llm.chat(
                messages, model=self.llm_model, temperature=0.7, max_tokens=1024,
                seed=self.seed * 1000 + n + seed_offset + 77,
                thinking=self.sim_thinking,
            )
            self.stats["llm_calls"] += 1
        parsed = _parse_protocol(result.text)
        if parsed is None:
            self.stats["json_failures"] += 1
            return None, ["输出不是合规 JSON 协议"]
        text = str(parsed.get("text") or "").strip()
        # 剥掉括号舞台指示（见 STAGE_DIRECTION_RE 的注释：音频侧被 TTS 吞掉无害，
        # 但文本侧会污染裁判/泄漏探针/人审）。计数进 stats，剥空则由下面「text 为空」拦下重试。
        _n_sd = len(STAGE_DIRECTION_RE.findall(text))
        if _n_sd:
            self.stats["stage_direction_stripped"] += _n_sd
            text = STAGE_DIRECTION_RE.sub("", text).strip()
        parsed["text"] = text
        # 内部方案（用户授权）：v7 值域中文化后，模型可能输出中文状态标签；按 v7 分支用
        # normalize_state 归一回英文 canonical 码，使下游 candidate_states 校验 / key_event 比对 /
        # _vocab_defense / TTS 查表契约完全不变。仅 v7 生效——v6/v6.1 渲染英文值域、模型输出
        # 英文码，此分支不进，行为字节不变（「改动只落 v7 路径」/ V8）。normalize_state 对
        # 英文码幂等（base.py:79-80），故模型即便仍输出英文码也安全；归一失败（垃圾值）保持原样，
        # 由下面的 candidate_states 校验照常拦下。
        if self.prompt_schema == "v7":
            _norm_state = normalize_state(str(parsed.get("state") or ""))
            if _norm_state:
                parsed["state"] = _norm_state
        problems: list[str] = []
        if not text:
            return None, ["text 为空"]
        if not (MIN_UTTERANCE_CHARS <= len(text) <= MAX_UTTERANCE_CHARS):
            problems.append(
                f"text 需 {MIN_UTTERANCE_CHARS}–{MAX_UTTERANCE_CHARS} 字（含标点），"
                f"当前 {len(text)} 字")
        if agent_reply and (text in agent_reply or agent_reply[:15] in text):
            problems.append("text 与客服上轮回复高度重合，只说用户自己的话")
        if parsed.get("state") not in self.candidate_states:
            problems.append(f"state 必须是 {'/'.join(self.candidate_states)} 之一")
        is_key = bool(parsed.get("is_key_turn"))
        if is_key and not self.key_completed and parsed.get("state") != self.key_event.get("state"):
            problems.append(
                f"关键轮 state 必须是 {self.key_event.get('state')}（关键事件指定状态）")
        end_call = bool(parsed.get("end_call"))
        if end_call and not (self.key_completed or is_key):
            problems.append("还不能挂断，你还有事没办（关键事件完成后才可道别收尾）")
        # covered 规范化（非法项静默丢弃，不计违规）
        covered: list[int] = []
        for c in parsed.get("covered") or []:
            try:
                ci = int(c)
            except (TypeError, ValueError):
                continue
            if 1 <= ci <= len(self.mission) and ci not in covered:
                covered.append(ci)
        parsed["covered"] = covered
        return parsed, problems

    def _vocab_defense(self, text: str, is_key_turn: bool) -> list[str] | None:
        """词表运行时防线。返回 None=合规；非空=违规命中词（已重试后保留）。

        T2/T3 全部轮 + T1 非关键轮：runtime_emotion_hits 零命中；
        T1 关键轮：命中 ≥1 个 key_event.state 子表词（正向校验；
        neutral 关键事件无子表词则不设要求）；
        T2 关键轮（K18c 新增）：除零命中外还需命中 T2 白名单下限。
        """
        from listen2serve.vocab import runtime_emotion_hits, state_pool, t2_whitelist_hit

        level = self.leakage_label
        if level in ("T2", "T3") or (level == "T1" and not is_key_turn):
            hits = runtime_emotion_hits(text)
            if hits:
                return hits
            # K18c：T2 关键轮接上白名单下限。validate_level_v2 早已定义该下限（须命中
            # 信号词或缓和句式），但运行期只做零命中，下限一直闲置，T2 因此退化为
            # “只禁不导”，与 T3 的差别仅剩结论禁令（K18b 出现 T2 均分反高于 T1 的倒挂）。
            # 该下限本身宽松（“这个/先/再”等常见词均计命中），作用在于拦住 T2 最典型的
            # 病灶——不带任何缓和词的短硬结论句（如“我不打算买”）；主力约束仍在 prompt 侧。
            if level == "T2" and is_key_turn and not t2_whitelist_hit(text):
                return ["未命中 T2 信号词/缓和句式（白名单下限）"]
            return None
        if level == "T1" and is_key_turn:
            pool = state_pool(str(self.key_event.get("state")))
            if pool and not any(w in text for w in pool):
                return [f"未命中 {self.key_event.get('state')} 子表词"]
        return None

    # ---- TTS instruct 查表（）----
    def _assemble_instruct(self, state: str, is_key: bool) -> str:
        """按 base_scenario_id + state 查预生成表；查不到即数据契约违约，硬失败。

        关键轮固定取 `key` 条（不分透明度），非关键轮取 `non_key[state]`——表的索引键
        不含 -T1/-T2/-T3，所以“同 base 同 state 三档 instruct 逐字一致”是结构上的必然，
        不靠代码约定维持。

        韵律中性臂（`prosody_arm="neutral"`， 步骤 4 的 A / C 两格）固定取本 base 的
        `non_key["neutral"]`，与本轮实际 state、是否关键轮全无关 —— 韵律通道恒为中性，
        而身份短语（年龄 / 角色）仍随 base 变，因为它是说话人特征、不是情绪信息。
        """
        entry = self._instruction_entry # 构造期已加载并校验，缺失即 raise
        if self.prosody_arm == "neutral":
            try:
                return str(entry["non_key"]["neutral"])
            except KeyError:
                raise ValueError(
                    f"中性臂缺 neutral 条：base={self.base_scenario_id} "
                    f"candidate_states={list(entry['candidate_states'])}"
                ) from None
        if is_key:
            return str(entry["key"])
        try:
            return str(entry["non_key"][state])
        except KeyError:
            raise ValueError(
                f"instruction 表缺 state：base={self.base_scenario_id} state={state} "
                f"candidate_states={list(entry['candidate_states'])}"
            ) from None

    def _forced_tag(self, state: str, is_key: bool) -> str:
        """本轮由表强制附加的标签（无则空串）。

        不交 LLM：压声类 hushed 的耳语音质纯文本做不到（b），而“韵律的唯一输入是
        state”要求它不能时有时无——交给 LLM “按需”的话，同一个 base 会时而耳语时而不是。

        中性臂下一律不附加：强制标签本身就是韵律信息，留着会把该臂污成“半中性”，那 41 个
        挂标签的 base（Step2 v6.4 均衡子集）的 A / C 两格就不再是干净对照。
        """
        if self.prosody_arm == "neutral":
            return ""
        entry = self._instruction_entry
        tag = entry.get("key_tag") if is_key else (entry.get("tags") or {}).get(state)
        return str(tag or "")

    def _tagged_text(self, text: str, state: str, is_key: bool) -> tuple[str, Any]:
        """先前置表驱动强制标签，再过四重校验；返回 (强制标签, TagResult)。

        顺序很重要：强制标签先拼到句首，于是 LLM 自行输出的标签自动变成“非句首”而被
        规则 2 剔掉；无强制标签的 130 个 base 则因允许集为空而被规则 1 剔掉。两者合起来
        就是“标签全由数据表确定性附加”（）。
        """
        forced = self._forced_tag(state, is_key)
        raw = f"[{forced}]{text}" if forced else text
        res = apply_tag_policy(raw, allowed=self._allowed_tags,
                               model=str(self.llm_model or ""), enable_rich=False)
        if res.stripped:
            self.stats["tag_stripped"] += len(res.stripped)
        return forced, res

    def _state_for_turn(self, parsed: dict[str, Any]) -> str:
        """LLM 声明的 state 为准；mission 声明的 state 用于交叉校验，不覆盖。

        一次演绎里 LLM 可能合理地比脚本晚一轮进入情绪，强行覆盖会造成“文本平静但声音
        愤怒”的错配（正是本次 bug 的镜像）。偏差进指标，由数据说话再定是否改为强制。
        关键轮的强口径不变（_call_and_validate 里以数据为准 + 记 state_mismatch）。
        """
        covered = parsed.get("covered") or []
        expected = self._mission_state_of(covered[-1]) if covered else self._last_state
        state = str(parsed.get("state") or "")
        if expected and state != expected:
            self.stats["state_drift"] += 1
            parsed["_state_expected"] = expected # 进 llm_debug，供 state 依从率统计
        return state

    def _mission_state_of(self, beat_no: int) -> str:
        """mission 第 beat_no 条（1 基）声明的 state；越界或省略则空串。"""
        if not (1 <= int(beat_no) <= len(self.mission_entries)):
            return ""
        return str(self.mission_entries[int(beat_no) - 1].get("state") or "neutral")

    # ---- TTS 合成 + 后处理 ----
    def _synthesize(self, text: str, state: str, tts_control: str) -> bytes:
        # 内部方案：送 TTS 前文本规范化（数字读音/多音字高危词）；
        # 只改合成输入，落盘文本与词表校验仍对原文。
        from listen2serve.runtime.tts_normalize import normalize_tts_text

        style = _instruct_from_control(
            state, tts_control, scenario_id=self.scenario_id, turn=self._turn_count + 1)
        prosody = _prosody_from_control(tts_control)
        result = self.tts.synthesize(normalize_tts_text(text), voice=self.voice,
                                     style=style, prosody=prosody)
        pcm = result.audio
        if self.audio_mode == "regular" or self.effect_config is not None:
            import numpy as np

            arr = np.frombuffer(pcm, dtype=np.int16)
            arr = apply_effects(
                arr, result.sample_rate,
                self.effect_config or EffectConfig(),
                np.random.default_rng(self.seed),
            )
            pcm = arr.tobytes()
        return pcm

    # ---- 演绎 prompt（单一模板；稳定前缀 + 动态尾部，/）----
    def _build_messages(self, n: int, agent_reply: str, *, forced: bool) -> list[dict[str, str]]:
        if self.prompt_schema == "v7":
            return self._build_messages_v7(n, agent_reply, forced=forced)
        if self.prompt_schema == "v6.1":
            return self._build_messages_v61(n, agent_reply, forced=forced)
        us = self.user_script
        identity = us.get("identity", "")
        goal = us.get("user_goal") or us.get("goal", "")
        style_block = _render_persona_style(us.get("persona_style"))
        anchor = _render_fact_anchor(self.db_seed)
        facts_block = _render_user_facts(us.get("user_facts"))
        disclosure_block = _render_disclosure_plan(self.disclosure_plan)
        mission_block = _render_mission(self.mission)
        covered = sorted(self._covered)
        progress = "、".join(str(c) for c in covered) if covered else "无"
        key_status = (
            f"已在第 {self._key_turn_no} 轮完成" if self.key_completed else "未完成")
        ke = self.key_event
        key_block = (
            f"【关键事件】{ke.get('description', '')}（要求状态：{ke.get('state')}，"
            f"触发时机：{ke.get('trigger_hint', '自然推进时')}；"
            f"当前状态：{key_status}）\n"
        )
        level = self.leakage_label or "T2"
        transparency = _transparency_clause(level, self._t1_sample_words)
        # ---- 轮次预算（软预算进 prompt + 强制条款）----
        budget_line = f"【轮次预算】整通电话争取在 {self.soft_budget} 轮内办完你要办的事；当前第 {n} 轮。\n"
        if not self.key_completed:
            if forced:
                budget_line += "【预算强制】本轮必须完成【关键事件】（is_key_turn=true）。\n"
            elif n >= self.soft_budget:
                budget_line += "【预算告警】关键事件尚未完成，请尽快收尾，最迟下一轮必须完成【关键事件】。\n"
        elif n >= self.soft_budget:
            budget_line += "【预算告警】事已办完，请尽快道别收尾（end_call=true）。\n"
        forbidden = us.get("forbidden_user_behavior", [])
        forbidden_block = _render_forbidden(forbidden)
        examples = _fewshot_examples(self.role, self.dynamics)
        fewshot_block = ""
        if examples:
            fewshot_block = (
                "【风格示范】以下口语短句仅作语气/节奏参考，严禁照抄内容：\n"
                + "\n".join(f"- {e}" for e in examples) + "\n"
            )
        history_block = _render_history(self._history)
        # 稳定前缀（system + 身份/风格/目标/事实/披露/mission/关键事件定义/透明度）
        # 与动态尾部（进度/预算/历史/客服上轮）分段，利于后端隐式前缀缓存。
        system = (
            "你正在与客服进行一通真实的中文电话通话，扮演打/接电话的用户。\n"
            "你只扮演【用户】一方：绝不能说客服的台词，绝不能重复或改写客服刚才说的话。\n"
            "每轮必须先承接客服上一轮的问题或话题（先回应、再推进），不得跳过客服的话题自说自话；\n"
            "收尾轮先用一短句回应客服刚提出的事项，再道别，不得说完自己的事立刻挂断；"
            "若客服还在登记必要信息（如姓名/联系方式/地址），先把信息给完再道别。\n"
            "每轮自然承接客服上文，只说用户自己的话：10–60字（含标点）的中文口语，"
            "只能说中文，不允许出现英文单词或字母；不要书面语、不要解释、不要列表。\n"
            "禁止自述式态度铺陈：像“我都听你的安排”“这些我都拿笔记下了”这类台词禁止出现。\n"
            "禁止纯语气词应付（单独的“嗯”“好的”不合格）；每轮须含实质信息"
            "（回应、追问、态度或新信息）；语气词与迟疑自我修复合计最多一处。\n"
            "只输出一个紧凑单行 JSON（不要 markdown 围栏、不要任何其他文字）：\n"
            '{"text":"你的台词","state":"情绪状态","is_key_turn":本轮是否执行【关键事件】,'
            '"end_call":是否道别收尾,'
            '"covered":["本轮完成的剧情要点序号"],"interrupt":是否以插话方式开口(仅关键轮可用)}\n'
            f"state 值域：{'/'.join(self.candidate_states)}。is_key_turn 全场有且仅有一次 true；"
            "end_call 只有在【关键事件】完成之后才允许 true。"
        )
        stable_prefix = (
            f"【身份】{identity}\n"
            f"{style_block}"
            f"【目标】{goal}\n"
            f"【事实锚】{anchor}\n"
            f"{facts_block}"
            f"{disclosure_block}"
            "【事实锚定硬约束】你话里涉及的一切业务事实只能来自【身份】【事实锚】与"
            "【用户本人掌握的事实】；严禁虚构任何新事实；上述来源未提供的事实不得给出"
            "具体数字或日期，可用‘大概’‘回头定了告诉你’等模糊表述。\n"
            "【事实锚使用边界】【事实锚】是系统侧记录：其中客服尚未亲口提到的细节"
            "（如产品价格、活动细则、订单内部状态等）你并不知情，绝不主动说出；"
            "等客服提到后你才可以确认、追问或讨价。\n"
            f"{mission_block}"
            f"{key_block}"
            f"【透明度约束】{transparency}\n"
            f"{fewshot_block}"
            f"【硬约束】\n{forbidden_block}\n"
        )
        dynamic_tail = (
            f"【进度】已完成要点：{progress}\n"
            f"{budget_line}"
            f"【完整对话历史】\n{history_block}\n"
            f"【客服上轮回复】{agent_reply or '（对话开始，无上轮）'}\n"
            "【要求】自然承接客服上文，按协议输出紧凑 JSON；covered 里报出本轮完成的要点序号。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": stable_prefix + dynamic_tail},
        ]

    # ---- 内部方案：四块式减负 prompt（v6.1 schema，A/B 验证后切默认）----
    def _build_messages_v61(self, n: int, agent_reply: str, *, forced: bool) -> list[dict[str, str]]:
        """【你是谁】【要办的事】【怎么说话】【输出协议】四块式：
        - 叙事式处境取代 11 个【】区块；事实锚退役（不知情信息物理移除，不给即不知）；
        - 负面清单压到 ≤4 条；透明度约束保留（词表防线不可破）。
        """
        us = self.user_script
        identity = us.get("identity", "")
        goal = us.get("user_goal") or us.get("goal", "")
        facts_block = _render_user_facts_narrative(us.get("user_facts"))
        consensus = self.background_consensus
        # 块一【你是谁】：叙事式一段话（姓名/处境 + 共识背景 + 这通电话对你意味着什么）
        who = f"【你是谁】{identity}"
        if consensus:
            who += ("" if str(identity).endswith(("。", "；")) else "。") + f"背景：{consensus}"
        if goal:
            who += f"这通电话对你来说：{goal}。"
        who += "\n" + (facts_block or "")
        # 块二【要办的事】：mission 要点 + 关键事件 + 披露时机（v6.0 语义保留）
        what = _render_mission(self.mission).replace("【剧情要点】", "【要办的事】", 1)
        ke = self.key_event
        key_status = (
            f"已在第 {self._key_turn_no} 轮完成" if self.key_completed else "未完成")
        what += (
            f"【关键事件】{ke.get('description', '')}（要求状态：{ke.get('state')}，"
            f"触发时机：{ke.get('trigger_hint', '自然推进时')}；"
            f"当前状态：{key_status}）\n"
        )
        what += _render_disclosure_plan(self.disclosure_plan)
        level = self.leakage_label or "T2"
        what += f"【透明度约束】{_transparency_clause(level, self._t1_sample_words)}\n"
        # 块三【怎么说话】：≤3 条（口语/字数/风格一句话）
        style_line = _persona_one_liner(us.get("persona_style"))
        how = (
            "【怎么说话】像真实打电话的普通人：中文口语，每轮 10–60 字（含标点），"
            "可以带自然的语气词与迟疑"
            + (f"；{style_line}" if style_line else "") + "\n"
        )
        # 块四【输出协议】：紧凑 JSON（内部方案 删 tts 槽位后 6 字段）+ ≤4 条硬约束
        protocol = (
            "【输出协议】只输出一个紧凑单行 JSON（不要 markdown 围栏、不要其他文字）：\n"
            '{"text":"你的台词","state":"情绪状态","is_key_turn":本轮是否执行【关键事件】,'
            '"end_call":是否道别收尾,'
            '"covered":["本轮完成的剧情要点序号"],"interrupt":是否以插话方式开口(仅关键轮可用)}\n'
            f"state 值域：{'/'.join(self.candidate_states)}。is_key_turn 全场有且仅有一次 true；"
            "end_call 只有在【关键事件】完成之后才允许 true。\n"
            "【硬约束】①不得复读或改写客服刚说的话；②不得自述式铺陈（如“我都记下了”）；"
            "③业务事实只能来自上面给你的信息，没有的不要编（可用“大概/回头定了告诉你”模糊）；"
            "④单独语气词（只回“嗯/好的”）不合格，每轮须有实质信息。"
        )
        # 动态尾部（与 v6.0 同构：进度/预算/历史/客服上轮）
        covered = sorted(self._covered)
        progress = "、".join(str(c) for c in covered) if covered else "无"
        budget_line = f"【轮次预算】整通电话争取在 {self.soft_budget} 轮内办完你要办的事；当前第 {n} 轮。\n"
        if not self.key_completed:
            if forced:
                budget_line += "【预算强制】本轮必须完成【关键事件】（is_key_turn=true）。\n"
            elif n >= self.soft_budget:
                budget_line += "【预算告警】关键事件尚未完成，请尽快收尾，最迟下一轮必须完成【关键事件】。\n"
        elif n >= self.soft_budget:
            budget_line += "【预算告警】事已办完，请尽快道别收尾（end_call=true）。\n"
        history_block = _render_history(self._history)
        system = (
            "你正在与客服进行一通真实的中文电话通话，扮演打/接电话的用户。\n"
            "你只扮演【用户】一方：绝不能说客服的台词。\n"
            "每轮先承接客服上一轮的问题或话题（先回应、再推进），只说中文口语。"
        )
        user = (
            who + what + how + protocol + "\n"
            + f"【进度】已完成要点：{progress}\n"
            + budget_line
            + f"【完整对话历史】\n{history_block}\n"
            + f"【客服上轮回复】{agent_reply or '（对话开始，无上轮）'}\n"
            + "【要求】自然承接客服上文，按协议输出紧凑 JSON；covered 里报出本轮完成的要点序号。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ---- 内部方案：prompt v7 叙事化（在 v6.1 四块式基础上三点增量）----
    def _build_messages_v7(self, n: int, agent_reply: str, *, forced: bool) -> list[dict[str, str]]:
        """v7 = v6.1 四块式 + 三点增量：
        ① 透明度约束并入关键事件一句【语气提示】（不再单列区块；词表防线兜底不变）；
        ② 历史压缩：近 4 轮全文 + 更早轮一句摘要（防长对话膨胀）；
        ③ few-shot 真实语料示范注入【怎么说话】（正面示范替代负面清单）。

        内部方案 逐样本审阅后的修正（仅 v7 生效，v6.x 文案冻结）：
        ① 关键轮透明度子句改用 _transparency_key_clause_v7（T3 态度陈述泄露）；
        ② 【语气提示】增“不提前摊牌”（关键事件被提前满足 → 关键轮变重复提问，KeyTurn 虚高）；
        ③ 披露时机改 narrative 文案（示例句被照抄成披露元话语）；
        ④ 关键事件描述显式标注“意图说明、不是台词”（mission 那条本身就把结论写在明面上，
          与 T3 “结论不上文本”相互矛盾）；
        ⑤ few-shot 按透明度过滤结论式示范句 + 引导语改“不学态度和内容”。
        """
        us = self.user_script
        identity = us.get("identity", "")
        goal = us.get("user_goal") or us.get("goal", "")
        facts_block = _render_user_facts_narrative(us.get("user_facts"))
        consensus = self.background_consensus
        # 块一【你是谁】：叙事式一段话（姓名/处境 + 共识背景 + 这通电话对你意味着什么）
        who = f"【你是谁】{identity}"
        if consensus:
            who += ("" if str(identity).endswith(("。", "；")) else "。") + f"背景：{consensus}"
        if goal:
            who += f"这通电话对你来说：{goal}。"
        who += "\n" + (facts_block or "")
        # 块二【要办的事】：mission 要点 + 关键事件 + 披露时机
        # 内部方案：【语气提示】不再塞进 what，移到动态尾部【本轮怎么说】（见本函数末）。
        what = _render_mission(self.mission).replace("【剧情要点】", "【要办的事】", 1)
        ke = self.key_event
        key_status = (
            f"已在第 {self._key_turn_no} 轮完成" if self.key_completed else "未完成")
        level = self.leakage_label or "T2"
        # 内部方案：description 拆 objective（中性业务目标，给演绎侧）+ intent（内心动机）。
        # S3 尚未改数据时带回落（objective/intent 均回落到 description），保证 S1 能跑通、能出
        # dump 页；S3 改完后回落分支保留供历史复算（接口约定见 内部方案 §改动3）。
        objective = ke.get("objective") or ke.get("description", "")
        intent = ke.get("intent") or ke.get("description", "")
        trigger = ke.get("trigger_hint", "自然推进时")
        if level == "T3":
            # 内部方案：T3 整个「要求状态」括号不渲染（不向演员揭示状态标签，堵 P2 泄露源），
            # 只给「这件事你心里怎么想」（intent）——态度只由韵律承载、文字读不出。
            what += (
                f"【关键事件】{objective}（触发时机：{trigger}；当前状态：{key_status}）\n"
                f"这件事你心里怎么想：{intent} —— 但这轮不许把这份想法说出口，只让语气带出来。\n"
                f"要点里带【关键事件】的那条、以及上面这句描述，都是给你看的意图说明，"
                f"不是台词，别照着念。\n"
            )
        else:
            # 内部方案：T1（及历史 T2）要求状态用中文标签，从 STATE_LABELS 取
            # （不许在 prompt 里另写一份标签，与 agent.py:45 同源纪律一致）。
            state_zh = STATE_LABELS.get(str(ke.get("state")), str(ke.get("state")))
            what += (
                f"【关键事件】{objective}（要求状态：{state_zh}，"
                f"触发时机：{trigger}；当前状态：{key_status}）\n"
                f"这件事你心里怎么想：{intent} —— 关键轮要把它说出来。\n"
                f"要点里带【关键事件】的那条、以及上面这句描述，都是给你看的意图说明，"
                f"不是台词，别照着念。\n"
            )
        # 内部方案：【本轮怎么说】= 原【语气提示】的压缩版（常驻原则句，无词表/例句）
        # + 关键轮那一次注入的 1 条正例（改动5：正例只在 forced 拼上，常驻位置不放例句）。
        key_clause = _transparency_key_clause_v7(
            level, self._t1_sample_words, ke.get("surface"), self.key_form)
        say_this = (
            "【本轮怎么说】平时（非关键轮）克制中性、不带情绪词，"
            "也不提前摊出关键事件那轮的态度或结论"
            f"{_pre_key_stance_ban_v7(level)}；"
            f"轮到关键事件那轮：{key_clause}"
        )
        if forced:
            say_this += _key_turn_positive_example(self.key_form, level)
        # 内部方案 的 A/B 仪器（红线4）：块序开关。默认 "new"=改动6 生效（【本轮怎么说】
        # 落在动态尾部、紧邻【客服上轮回复】之后）；"old"=放回改动6 之前的位置（what 块中段，
        # 原【语气提示】处），供 A/A+A/B 隔离「块序」这一个变量。V7 验收后保留作复现仪器。
        block_order = os.environ.get("SPEECH_BENCH_S1_BLOCK_ORDER", "new")
        if block_order == "old":
            what += say_this + "\n"
        what += _render_disclosure_plan(self.disclosure_plan, narrative=True)
        # 块三【怎么说话】：口语/字数 + persona 或默认句 + 真实语料 few-shot 正面示范
        style_line = _persona_one_liner(us.get("persona_style"))
        examples = _filter_examples_for_level(_fewshot_examples(self.role, self.dynamics), level)
        fewshot_lines = ""
        if examples:
            fewshot_lines = (
                "像真实通话里那样说话，比如这种语气节奏（只学语气和口语习惯，"
                "不学它们的态度和内容）：\n"
                + "\n".join(f"- {e}" for e in examples) + "\n"
            )
        how = (
            "【怎么说话】像真实打电话的普通人：中文口语，多数轮 10–30 字（含标点），"
            "最长别超过 45 字；只是应一声、认个账或答应个时间的轮次，几个字就行。"
            "可以带自然的语气词与迟疑"
            + (f"；{style_line}" if style_line else "") + "\n"
            + fewshot_lines
        )
        how += _realism_block(self.role, self.role_type, self.dynamics)
        # 块四【输出协议】：紧凑 JSON（内部方案 删 tts 槽位后 6 字段）+ ≤4 条硬约束
        protocol = (
            "【输出协议】只输出一个紧凑单行 JSON（不要 markdown 围栏、不要其他文字）：\n"
            '{"text":"你的台词","state":"情绪状态","is_key_turn":本轮是否执行【关键事件】,'
            '"end_call":是否道别收尾,'
            '"covered":["本轮完成的剧情要点序号"],"interrupt":是否以插话方式开口(仅关键轮可用)}\n'
            # 内部方案（用户授权）：v7 值域中文化，消除中文演员 prompt 里的英文状态码
            # （P1「这是个实验」的元信号）。模型输出的 state 在 _call_and_validate 里按 v7 分支
            # 用 normalize_state 归一回英文 canonical 码，下游 candidate_states 校验 / key_event
            # 比对 / TTS 查表契约不变；v6/v6.1 仍渲染英文值域、行为字节不变（/V8）。
            # 用全角「／」连接，避免与 doubtful 标签「疑问/不信」内的半角斜杠混淆（与 agent.py:48 同源）。
            f"state 值域：{'／'.join(STATE_LABELS.get(s, s) for s in self.candidate_states)}。is_key_turn 全场有且仅有一次 true；"
            "end_call 只有在【关键事件】完成之后才允许 true。"
            "end_call=true 的那一轮，台词本身必须把收尾的意思说出口 —— 明确表示你这边"
            f"没别的事了 / 就这样 / 要挂了（例如{_end_call_examples(level)}），"
            "可以先答完客服上一问再收；不能只是接着陈述，更不能再抛新"
            "问题。你不说出口，客服就无从知道这通电话要结束了。\n"
            "【硬约束】①不得原样照搬客服的话（可以用自己的口语把关键信息复述确认，"
            "但不许逐字段罗列成念表格，也不许在确认前加“你们这安排挺明白”之类的评价套话）；"
            "②不得自述式铺陈（如“我都记下了”）；"
            "③业务事实只能来自上面给你的信息，没有的不要编（可用“大概/回头定了告诉你”模糊）；"
            "④不合格的是**空轮**而不是短轮：只有语气词、没落到任何具体内容上的轮次不行"
            "（光回一个“嗯”“好的”就属这种）；短轮本身没问题 —— 只要这几个字落在一个具体的点上"
            "（应下一个金额、一个时间、一个渠道，或认下一件事），就算合格。"
        )
        # 动态尾部：进度/预算/压缩历史/客服上轮
        covered = sorted(self._covered)
        progress = "、".join(str(c) for c in covered) if covered else "无"
        budget_line = f"【轮次预算】整通电话争取在 {self.soft_budget} 轮内办完你要办的事；当前第 {n} 轮。\n"
        if not self.key_completed:
            if forced:
                budget_line += "【预算强制】本轮必须完成【关键事件】（is_key_turn=true）。\n"
            elif n >= self.soft_budget:
                budget_line += "【预算告警】关键事件尚未完成，请尽快收尾，最迟下一轮必须完成【关键事件】。\n"
        elif n >= self.soft_budget:
            budget_line += "【预算告警】事已办完，请尽快道别收尾（end_call=true）。\n"
        if self.key_completed:
            # K18b 复跑发现：T3 关键轮改成中性句后，模型觉得“没演够”，于下一轮把结论补说
            # 并重新自报 is_key_turn=true（key_downgrade 由1→6 次），关键轮信息量被摊到两轮。
            budget_line += (
                "【关键事件已完成】那个意思你上面已经表达过了，后面几轮顺着往下走"
                "（问清后续、确认细节或道别），不要回头把同一个态度或结论再说一遍，"
                "is_key_turn 一律填 false。\n"
            )
        history_block = _render_history_compressed(self._history)
        system = (
            "你正在与客服进行一通真实的中文电话通话，扮演打/接电话的用户。\n"
            "你只扮演【用户】一方：绝不能说客服的台词。\n"
            "每轮先承接客服上一轮的问题或话题（先回应、再推进），只说中文口语。\n"
            "收尾也要收得自然：客服上一轮问了你什么、或还在登记你的信息（姓名、"
            "联系方式、地址、时间），先把这一问答完再道别，别把话丢在半空就挂；"
            "事情办完了不必抢着结束，让客服把该说的说完 —— 但你真决定要收尾的那一轮，"
            "得把收尾的意思说出口，别让客服猜。"
        )
        user = (
            who + what + how + protocol + "\n"
            + f"【进度】已完成要点：{progress}\n"
            + budget_line
            + f"【对话】\n{history_block}\n"
            + f"【客服上轮回复】{agent_reply or '（对话开始，无上轮）'}\n"
            # 内部方案：new 序时【本轮怎么说】落在这里（紧邻客服上轮回复之后、要求之前）；
            # old 序（改动6 的 A/B 仪器）时它已在 what 块中段，此处不重复注入。
            + (say_this + "\n" if block_order != "old" else "")
            + "【要求】自然承接客服上文，按协议输出紧凑 JSON；covered 里报出本轮完成的要点序号。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


# ---- 协议解析 ----
def _parse_protocol(text: str) -> dict[str, Any] | None:
    """从 LLM 输出提取协议 JSON（容忍 markdown 围栏/前后杂散文字）。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---- prompt 小节渲染 ----
def _render_persona_style(persona_style: list[Any] | None) -> str:
    """persona 风格卡（口头禅频率封顶条款保留）。"""
    if not persona_style:
        return ""
    items = [str(s) for s in persona_style if str(s).strip()]
    if not items:
        return ""
    return (
        f"【说话风格】你的说话风格：{'；'.join(items)}。风格服从于透明度约束与剧情需要，"
        "不得因风格添加情绪词（优先级：文本约束/词表 > 剧情 > 说话风格）。"
        "其中口头禅类习惯整场对话自然地用至多 1 次：前面轮次已用过就不再用；"
        "禁止用在你的第一句发言里；禁止连续多轮用同一个词开头。\n"
    )


def _render_fact_anchor(db_seed: dict[str, Any] | None) -> str:
    if not db_seed:
        return "（无额外事实锚，仅使用【身份】中给出的业务事实）"
    lines: list[str] = []
    for table, rows in db_seed.items():
        for row in rows or []:
            if isinstance(row, (list, tuple)):
                lines.append(f"{table}: " + " | ".join(str(x) for x in row))
            else:
                lines.append(f"{table}: {row}")
    return "\n".join(lines) if lines else "（无额外事实锚，仅使用【身份】中给出的业务事实）"


_USER_FACT_LABELS: dict[str, str] = {
    "planned_amount": "拟还金额",
    "planned_date": "拟还日期",
    "income_note": "收入与资金安排",
    "issue_detail": "问题情况补充",
    "budget_or_concern": "预算与顾虑",
    # 内部方案（F7）：identity 里的处境类数字外迁到 user_facts，需要中文标签，
    # 否则 _render_user_facts_narrative 会把英文键名直接印进【你是谁】。
    # 年龄不外迁（gen_tts_instructions._IDENTITY_RE 要 identity 里有「NN 岁+职业」结构，
    # 它派生的 identity_phrase 已烘进冻结的 tts_instructions.json）。
    "debt_amount": "欠款金额",
    "overdue_days": "逾期天数",
    # 下面两个键在数据里早已存在（第三方寻人 2 base），此前一直没标签，
    # 渲染成「relation_to_debtor：债务人的老乡」这种中英混排；内部方案 要动这两条，顺手补上。
    "relation_to_debtor": "与债务人的关系",
    "relay_note": "转达要点",
}

_ZH_DIGITS = "零一二三四五六七八九"
_ZH_SEG_UNITS = ("千", "百", "十", "")


def _zh_below_wan(n: int) -> str:
    out = ""
    pending_zero = False
    for power, unit in zip((1000, 100, 10, 1), _ZH_SEG_UNITS):
        digit = n // power % 10
        if digit == 0:
            pending_zero = True
            continue
        if pending_zero and out:
            out += "零"
        pending_zero = False
        out += _ZH_DIGITS[digit] + unit
    return out


def amount_to_spoken_zh(value: float | int) -> str:
    """把金额转成口语中文。

    渲染成阿拉伯数字时 LLM 口语化会丢位：`10500` 说成"一万零五"而不是"一万零五百"。
    整数位一律预先转中文，让模型照读；小数按"点X"逐位读，符合口语习惯。
    """
    negative = value < 0
    value = abs(value)
    whole = int(value)
    frac = round(value - whole, 2)
    if whole == 0:
        head = "零"
    else:
        parts: list[str] = []
        yi, rest_yi = divmod(whole, 10**8)
        wan, rest = divmod(rest_yi, 10**4)
        if yi:
            parts.append(_zh_below_wan(yi) + "亿")
        if wan:
            if yi and wan < 1000:
                parts.append("零")
            parts.append(_zh_below_wan(wan) + "万")
        if rest:
            if (yi or wan) and rest < 1000:
                parts.append("零")
            parts.append(_zh_below_wan(rest))
        head = "".join(parts)
        # 口语里"一十五"说"十五"、"一十万"说"十万"，段首的"一十"要去掉那个"一"
        head = head.replace("一十", "十", 1) if head.startswith("一十") else head
    if frac:
        digits = f"{frac:.2f}".split(".")[1].rstrip("0")
        head += "点" + "".join(_ZH_DIGITS[int(d)] for d in digits)
    return ("负" if negative else "") + head


def _fact_text(key: str, value: Any) -> str:
    if key == "planned_amount" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{amount_to_spoken_zh(value)}元"
    return str(value)


def _render_user_facts(user_facts: dict[str, Any] | None) -> str:
    if not user_facts:
        return ""
    lines = []
    for key, value in user_facts.items():
        label = _USER_FACT_LABELS.get(key, key)
        lines.append(f"- {label}：{_fact_text(key, value)}")
    return "【用户本人掌握的事实】\n" + "\n".join(lines)


def _render_user_facts_narrative(user_facts: dict[str, Any] | None) -> str:
    """内部方案（v6.1 schema）：用户底牌叙事化（融入【你是谁】，不再列表化）。"""
    if not user_facts:
        return ""
    parts = []
    for key, value in user_facts.items():
        label = _USER_FACT_LABELS.get(key, key)
        parts.append(f"{label}：{_fact_text(key, value)}")
    return "你自己知道的实际情况：" + "；".join(parts) + "。\n"


def _persona_one_liner(persona_style: list[Any] | None) -> str:
    """内部方案（v6.1 schema）：persona 只留一句话轻特征（口头禅类随数据侧置空退役）。"""
    if not persona_style:
        return ""
    items = [str(s).strip() for s in persona_style if str(s).strip()]
    if not items:
        return ""
    return "你的说话习惯：" + items[0] + "（整通电话自然带出至多一次）"


def _render_disclosure_plan(plan: dict[str, Any] | None, *, narrative: bool = False) -> str:
    """【披露时机】：底牌字段逐一注明何时透露（治病灶三：抢跑/掀底牌）。

    narrative（内部方案 审阅修正，v7 专用）：原文案带的示例句被模型逐字照抄，产出
    “你问我还款能力，实际情况是…”这类披露元话语（真人不这么说话）；改为只讲要求不给句式。
    v6.0/v6.1 文案逐字冻结（A/B 可比）。
    """
    if not plan:
        return ""
    lines = []
    for key, when in plan.items():
        label = _USER_FACT_LABELS.get(key, key)
        lines.append(f"- {label}：{when}")
    if narrative:
        return (
            "【披露时机】以下信息是你的底牌，时机未到绝不主动透露，被问到且时机已到才说；"
            "说的时候就像聊家常一样直接说情况，不要复述客服问了什么，也不要凭空抛出数字：\n"
            + "\n".join(lines) + "\n"
        )
    return (
        "【披露时机】以下信息是你的底牌，时机未到绝不主动透露，被问到且时机已到才说；"
        "透露时先用半句话承接客服上一轮（如‘你问我还款能力，实际情况是…’），不得凭空抛出数字：\n"
        + "\n".join(lines) + "\n"
    )


_AFTER_HINT = {
    "agent_states_purpose": "等客服说明来意之后再做这一条，别自己先把话题点出来",
}


def _render_mission(mission: list[str] | list[dict[str, Any]]) -> str:
    """渲染剧情要点；`after` 字段渲染成要点行后的括注。

    内部方案：审计 144 base 查出 21 条 `mission[0]` 把**只有客服说了才知道的事**
    当成用户的已知前提 —— 来电主题（「先问清是哪栋的**物业费**」）、对方机构
    （「先核实对方**物业公司**身份」）、价格档位（「听对方介绍**3980 元档**」）。
    外呼电话是对方打来的，用户接起时并不知道这些，照着要点演就成了抢跑。

    修法是给这一拍标 `after`，**不是删掉要点** —— 要点表达的是场景设计意图
    （这个用户确实要问哪栋、确实要核实对方是谁），错的只是时机。

    `after` 必须在这里真的渲染出来：只往 `scenarios.jsonl` 加字段而运行期不认，
    数据看着改好了、送进模型的 prompt 一个字没变，是比不改更坏的结果。
    """
    lines = []
    for i, m in enumerate(mission, 1):
        beat = m if isinstance(m, str) else str(m.get("beat") or "")
        after = "" if isinstance(m, str) else str(m.get("after") or "")
        hint = _AFTER_HINT.get(after, "")
        lines.append(f"{i}. {beat}（{hint}）" if hint else f"{i}. {beat}")
    return "【剧情要点】（整通电话你要办成的事，顺序可自行调整，无需逐条念出）\n" + "\n".join(lines) + "\n"


def _transparency_key_clause(level: str, t1_words: list[str]) -> str:
    """透明度约束的关键轮分句（v6.0/v6.1 口径，逐字冻结）。"""
    if level == "T1":
        if t1_words:
            sample = "/".join(t1_words)
            return f"须自然带出至少 1 个该状态的明确情绪词（参考词：{sample}）。"
        return "按关键事件要求状态自然表达。"
    if level == "T3":
        return "仍用纯业务中性词汇表达，禁止任何情绪/态度词。"
    return "仅允许轻微信号词（那个/就是/能不能/嗯…），仍禁明确情绪词。"


def key_form_for(scenario_id: str) -> str:
    """关键轮句式（question / statement），按 base 场景号奇偶确定性配平。

    内部方案（2026-09-01）：planQ Step 1 的文字对照证明 T3 关键轮台词并非情绪中性 ——
    36 条里 33 条（91.7%）是问句，五个被测模型有 45.8%–72.2% 的应答都是 doubtful，
    而**只给文字不给音频**时准确率 27.8% 与给音频完全相同、doubtful 占比 64%，
    仅问句项的 doubtful 率 67%。即：问句这一句式本身就是个 doubtful 先验，
    韵律还没上场就已被文字盖住。故把句式作为需要配平的干扰因子处理。

    选择配平而非全改陈述句：陈述句同样携带先验（平铺直叙 → neutral），
    全改只是把 doubtful 偏置换成 neutral 偏置，B−C 依旧测不出韵律。

    奇偶而非 seed 哈希：句式必须在 T1/T2/T3 三层间逐条一致，否则层间可比性被破坏；
    base 号在同一 state 内近似连续，奇偶即得同态内近似对半（cooperative 36/36、
    displeased 24/19、doubtful 8/11、hushed 3/2、urgent 1/4）。
    """
    base = re.sub(r"-T[123]$", "", scenario_id or "")
    m = re.search(r"(\d+)$", base)
    if not m:
        return "question"
    return "statement" if int(m.group(1)) % 2 == 0 else "question"


def _transparency_key_clause_v7(
    level: str, t1_words: list[str], surface: str | None = None,
    form: str = "question",
) -> str:
    """v7 关键轮透明度子句（内部方案 审阅修正）。

    修正的偏离：T3 原句“纯业务中性词汇”只禁了情绪词，模型依旧直白说出态度结论
    （“我不打算购买这个产品”“实在等不起”）——态度陈述不在词表内，防线拦不住，
    于是 T3 实际与 T1 同等透明（plan “真意只由韵律承载”的操纵失效）；
    此处改为直接规定表面说法（中性代理动作）+ 给禁用句式清单。

    纸面二次校验修正（prompt 自相矛盾，未跑先改）：
    - 原“复述一下对方刚说的关键信息”与【硬约束】①“不得复读或改写客服刚说的话”
      直接冲突，删除该项；
    - 原“话里一个字也不许漏”歧义（漏＝泄露还是漏掉），改为“结论一个字都不许说出口”。

    K18b 复跑后补的第三处（过度矫正修正）：只禁结论、不限取材方向，模型为了“中性”
    挑了一句与自身意图相反的业务问句（MAR-ALL-23-T3 关键轮 displeased 态却问
    “那 99 元那个产品具体是咋用的呀？”，反倒像有兴趣），弦外之音被抹平且裁判据此
    误判客服“回避不满”。故补“同向约束”：中性话必须顺着自己的立场问下去，
    不得转成对对方提议感兴趣或答应下来。

    K18c 两处（单调性修正）：
    - 接入数据侧 surface（key_event.surface，分层表面动作）。此前三层共用同一条结论式
      description，T3 的“结论→中性句”翻译只能由模型即兴完成，K18b 三种翻译失败方式
      （结论后移／退化为无关问句／只演一半）全部出现；改由数据侧预先写好、可人工审，
      模型只需照说。缺该字段时回落原通用话术，保证未改写的场景照旧可跑。
    - T2 补正向下限。原 T2 子句只有“允许轻信号词 + 禁情绪词”，不规定该说什么，等于
      “T3 减去结论禁令”——态度结论可完整说出，实际文本透明度可能不低于 T1
      （K18b 中 T2 4.167 > T1 3.667 倒挂）。现要求必须带缓和句式且“结论不说全”，
      使三层泄露量单调：T1 说全、T2 说一半、T3 不说。
    - 同向约束改为立场无关表述。上一版写成“不能改口说成对对方的提议有兴趣、想了解
      更多或答应下来”，隐含假定关键事件立场必为负向；对正向关键事件（cooperative／
      satisfied，如“转为配合、愿意按方案办”）恰好把该场景的正确方向禁掉了。

    K18d 两处（层间区分度修正，K18c 逐样本审阅后补）：
    - T3 禁“立场铺垫句”。K18c 的 T3 结论禁令生效了，但模型改从**结论以外**的地方漏
      立场：在中性问句前加一句自述经历或让步语气词（MAR-ALL-23-T3“你们这种电话我拉黑
      好几个了，后面还会再打过来吗”探针 3 分；HOT-NEG-44-T3“行吧…要准备啥”探针 4 分）。
      这类句子既不含情绪词也不含结论词，词表与结论清单都拦不住，只能在 prompt 里
      按“句子结构”限死：T3 只许一句，且不许铺垫经历/感受，不许用表态度的语气词开头。
    - T2 补“可推断线索”下限。K18c 的 T2 与 T3 在 StateGuessAcc（均 33%）与
      StanceMarkers（均 0）上完全无法区分，只剩 leakage_score 差 1 分——因为“结论说一半”
      在模型手里退化成“干脆不说”，T2 事实上滑向 T3。T2 的设计定位是 implicit_consistent
      （隐性但一致），应当**可被推断**：故要求字面把态度指向一个具体的点（哪件事/哪个
      数字/哪个时间点让你有想法），使只读文字的人能猜出偏向，但仍不出现结论词。

    内部方案（2026-08-29，语用合理性）：
    - ❌ **T3 句子结构放宽已作废并回滚，本段保留作负面结论留痕。** K18d 的"只说这一句、
      不许任何铺垫"确实把 T3 逼成只能吐一个干问句（内部方案 同一病灶），内部方案
      因此提出放宽，并**预先立下硬门：T3 的 `leakage_score` 不许退化，退化即回滚**。
      两次探针（同 6 base，基线 `n7_20260826_qa3plus` T3=1.00）：

        · 第一版"最多两句 + 例词清单（嗯/那个/这个/我看一下）"
          → `planN_pragmatics_20260829` T3=**1.17**；且填充词起头率 24%→**92%**，
          "嗯"成逐轮口癖。根因：本子句常驻 system prompt（每轮都在模型视野里），
          **例词清单被当成推荐模板**而非一次性许可。
        · 第二版撤掉例词、只留"最多两句"
          → `planN_nofiller_20260829` T3=**1.50**，比第一版更差。
          `COL-ALL-04-T3` 关键轮："这笔钱我核实过了，**没想拖**。您说下具体怎么还、
          最晚啥时候到账？" —— 判官给 4 分且**猜中 cooperative**。

      机理：**"两句"等于给模型开了第二个句位，而模型拿它装立场陈述。** 一个干问句之所以
      不泄漏，恰恰因为它没有多余的位置放态度。所以 K18d 的单句禁令不是过度矫正，
      它是 T3 无线索的**承重结构** —— B−C 成立的前提，自然度不值得拿它换。
      结论：恢复 K18d 原措辞，`tests/内部测试用例` 反向断言防再次放宽。
    - ✅ T2 删元指令（保留）。原句"让只看文字的人能猜出你大致偏向哪边"等于告知演员有个
      纯文本判官在看，演员于是宁可说别扭的话也要塞线索（planL）。改为从角色自身
      视角陈述动机，"指向一个具体的点"这个可操作要求不变。T2 实测 2.67→2.50，未触发门槛。

    内部方案（2026-09-01，T3 句式配平）：T3 分支按 `form`（见 `key_form_for`）分岔。
    动机是 planQ Step 1 的文字对照：T3 关键轮 33/36 是问句，而**只给文字不给音频**时
    模型的 SER 准确率与给音频时相同（均 27.8%）、doubtful 占比 64% —— 问句这一句式
    自带 doubtful 先验，把韵律通道整个盖住，`prosody_only` 名不副实。
    单句禁令逐字保留（陈述句分支也只许一句），改的只是这一句的句式，没有开出第二个句位。
    数据侧 `surface` 写的是"只确认…"，语义上偏向询问，故陈述句分支明写"句末不许用问号"
    以压过 head 的倾向。也刻意不给例句 —— 本子句常驻 system prompt，
    内部方案 已证例词会被当成逐轮模板。
    注意上面 内部方案 那条"退化即回滚"的硬门**已被 内部方案 自己推翻**：三臂读数
    量出 `leakage_score` 的噪声底线 `mean|A2−A1|`=1.33（同一份数据复跑，12 条均值
    1.50→2.50），而 据以回滚的 1.00→1.17→1.50 整条落在噪声之内。现行判据是
    台账 A：凡读 `leakage_score` 的 A/B 差必须同批带 A/A 臂，`|B−A1|` 明显超过
    `|A1−A2|` 才算退化 —— 本次改动的验收照此办，判据见本函数 docstring。
    """
    # ---- 内部方案：T1/T3 拆成两个独立函数，本函数降为薄分发器（按 level 转发）。
    # 拆完 T1/T3 各自写自己的句子，不再共享任何跨 level 的句子（此前 T1/T3 共用 head=f"{surface}。"，
    # 现各自写死）。历史与偏离留痕仍在上面这段 docstring（K18b/c/d、内部方案、planR）。
    if level == "T1":
        return _key_clause_t1(t1_words, surface)
    if level == "T3":
        return _key_clause_t3(surface, form)
    # ---- T2：已停用（内部方案：透明度只有 T1/T3 两档；裁定6 砍 T2，依据 K18c 实测
    # T2 与 T3 在 StateGuessAcc 均 33%、StanceMarkers 均 0 不可区分）。分支保留仅供历史
    # run 复算，不要删。以下为冻结的 T2 原文，逐字不动。
    head = f"{surface}。" if surface else ""
    return (
        f"{head}只允许轻微信号词（那个/就是/能不能/嗯…）透一点口风，"
        "并且必须带上至少一个这类词或缓和句式（能不能/是不是/要不/先…），"
        "让人听出你有保留；不说明确情绪词，也不把态度结论说全（说一半、留个口子），"
        "态度主要靠语气。"
        "但要留一个能被看出来的线索：把你的想法指向一个具体的点"
        "（具体是哪件事、哪个数字、哪个时间点让你有想法），"
        "就像真打电话时那样——心里有想法，当着对方不把话说全，只把它挂在这一个点上。"
    )


def _key_clause_t1(t1_words: list[str], surface: str | None) -> str:
    """内部方案：T1 关键轮子句（从 _transparency_key_clause_v7 拆出，独立写死）。

    T1 = 文本明说上界对照（内部方案：T1 只跑 C 臂）。要求把态度说出来、至少带一个情绪词。
    历史与偏离留痕见 _transparency_key_clause_v7 的 docstring。改动1 后 T1 不再与 T3 共享句子。
    """
    if surface:
        tail = (
            f"其中至少带上一个情绪词（比如{'/'.join(t1_words)}）。" if t1_words
            else "语气和用词都要到位。"
        )
        # 与 _key_clause_t3 同：surface 自带尾句号时不再补，避免「。。」（数据句号变体健壮）
        head = surface if surface.endswith(("。", "？", "！", "?", "!")) else f"{surface}。"
        return f"{head}{tail}"
    if t1_words:
        sample = "/".join(t1_words)
        return f"把这份情绪说出来，自然带上至少一个情绪词（比如{sample}）。"
    return "把这份情绪说出来，语气和用词都到位。"


def _key_clause_t3(surface: str | None, form: str) -> str:
    """内部方案：T3 关键轮子句（从 _transparency_key_clause_v7 拆出并压缩）。

    T3 = 业务中性、态度只走韵律（主张量 B−C 的操纵）。改动4 把否定/禁止 token 从改前
    22–28（plan 散文记为 41，实测口径差异见报告）压到 ≤15：删掉枚举词表（依据 内部方案
    实测「例词清单被当成推荐模板」，填充词起头率 24%→92%；词表防线在代码侧 _vocab_defense /
    _T3_STANCE_MARKERS 仍生效， 不动），同向约束从双向举例压成一句、命令口气并入正例。
    **保留三条承重结构**（user_simulator.py:1331-1349 两次实测证明放宽即泄露）：
      ① 单句禁令「另外只说这一句」——两句等于开第二个句位、模型拿它装立场陈述；
      ② 陈述句分支的句式禁令（句末不许问号、不许能不能/是不是/吗/呢）——planR 句式配平；
      ③ 不许铺垫经历/感受、不许用表态度语气词开头——K18d 实测词表与结论清单都拦不住。
    正例改由改动5 在关键轮动态注入（_key_turn_positive_example），此处不放任何例句/词表。
    """
    ask = form != "statement"
    # 内部方案 V1 审阅修正：部分 surface 数据自带尾句号（如 MAR-ALL-06「…不表态、不评价。」），
    # 原 head=f"{surface}。" 会渲染成「。。」双句号；surface 是 S3 canonical 台词的蓝本（），
    # 双句号会 1:1 传导到台词，故对已带终止标点的 surface 不再补句号（对数据句号变体健壮）。
    if surface:
        head = surface if surface.endswith(("。", "？", "！", "?", "!")) else f"{surface}。"
    else:
        head = ("话里只说中性的业务句子（问一个细节、问下一步怎么办）。" if ask
                else "话里只说一句中性的业务事实（你记的金额/时间/做法）。")
    if ask:
        form_line = "另外只说这一句（一个问句）；"
    else:
        # 改动4 #2：陈述句句式禁令（承重结构）；命令/催办口气的 ban 已并入改动5 的正例示范。
        form_line = (
            "另外只说这一句，而且要说成陈述句：句末不许用问号，"
            "也不许用“能不能/是不是/吗/呢”这类问法；"
        )
    return (
        f"{head}态度只用语气传达，结论一个字都不许说出口，也不说任何情绪词；"
        "这句中性话必须顺着你在【关键事件】里的立场，不能说成反方向。"
        f"{form_line}前面不要铺垫自己的经历、感受或以往遭遇，"
        "也不要用表态度的语气词开头。"
    )


def _end_call_examples(level: str) -> str:
    """`end_call=true` 那轮的收尾示范句，**按透明度分档**（2026-09-06 内部方案 裁定 7 的同源扩展）。

    原来这里是写死的三条：「那就先这样吧」「行，我知道了，谢谢」「没别的事了」，
    对 T1/T2/T3 一律注入。问题是前两条**都是承接式接受**（「那就」「行，」正是 内部方案 闸门
    G7 的特征串，`hygiene.T3_ACCEPTANCE_PATTERNS`），而 T3 的整个设计是「态度只走语气、
    结论一个字都不许说出口」⇒ **prompt 在教 T3 的模型说一句闸门必定判死的话**。
    实测代价：canonical 小样 v3 仅剩的 2 条手改队列里，`HOT-NEG-48-T3`（cooperative/statement，
    intent 正是「想收尾」）四次尝试全部以「行，」开头并带「就按这个来」，四次全撞 G7。

    与 `_key_turn_positive_example` 是同一类缺陷、同一个修法：**T1 保留带态度的示范**
    （T1 的设计就是态度在文本里），**T3 换成读不出态度的中性收尾**。
    ⚠️ 这里沿用 内部方案 的实测结论 —— 例词清单会被演绎模型当成推荐模板
    （填充词起头率 24%→92%），所以示范句的形状**就是**产出台词的形状，不能只当注释看。
    由 `hygiene.t3_positive_example_violations()`（CI 断言 19）连同关键轮正例一起守着。
    """
    if level == "T3":
        # 中性收尾：只表达「这通电话到此为止」，不表达接受/拒绝/感谢任何一种结论
        return "“没别的事了”“就这样吧，我挂了”“先到这儿，我挂了啊”"
    return "“那就先这样吧”“行，我知道了，谢谢”“没别的事了”"


def _key_turn_positive_example(form: str, level: str) -> str:
    """内部方案：关键轮那一次注入的 1 条正例（示范台词），按 key_form 分问句/陈述句两版。

    常驻位置不放例句（内部方案 实测例词会被当成逐轮推荐模板），故正例只在 forced/关键轮拼上，
    且只给 1 条、只给正例不给反例（反例约束已在 _key_clause_t3 的三条承重结构里）。

    已知待办：正例取材应来自 S3 修好后的 canonical 关键轮台词（逐 base 挑选、人工审），
    现为手写占位正例（每个 key_form × 透明度各 1 条），S3 产出 canonical 台词后替换。
    """
    if level == "T1":
        # T1 关键轮要把态度说出来：正例示范「带情绪词、把结论说出口」
        if form != "statement":
            return "（示范·问句，仅参考语气句式、别照抄内容）比如：“这事到底靠不靠谱啊？我心里有点没底。”"
        return "（示范·陈述句，仅参考语气句式、别照抄内容）比如：“这个我挺满意的，那就这么定了吧。”"
    # T3 关键轮：态度只走语气、文字中性、只说一句（正例本身必须是读不出态度的中性业务句）
    if form != "statement":
        return "（示范·问句，仅参考语气句式、别照抄内容）比如：“那这个具体是怎么算的呀？”"
    # ⚠️ 陈述句这条**换过一次**（2026-09-06 内部方案 裁定 7，原句留档在下面）。
    # 原句：「那就按你说的这个时间来办。」—— 它是「接受并承诺执行」，也就是 cooperative 的结论，
    # 与本函数 docstring 自己写的「正例本身必须是读不出态度的中性业务句」直接矛盾；
    # 且一条占 内部方案 闸门 G7（承接式结论）的 **3 个**特征串（那就/就按/按你说的），
    # 实测在 **202/202 条 T3 场景**上撞闸（取数点：内部方案b 与
    # `（未发布）`）。因为 内部方案 实测过
    # 「例词清单被演绎模型当成推荐模板」（填充词起头率 24%→92%），正例的形状就是产出台词的形状 ——
    # canonical 小样 v2 的 G7 命中 9 次里 **4 次是这条正例的形状**（全在 HOT-NEG-48-T3，四次尝试全中）。
    # 新句在同样 202 条 T3 场景上跑八道闸**全过**，且零命中 `_T3_STANCE_MARKERS`、
    # `t3_action_ban.json` 的 ban 层（全态）、G7/G8 特征串；单句、零逗号、零阿拉伯数字、
    # 三个业务域都读得通。由 `hygiene.t3_positive_example_violations()`（CI 断言 19）守着不许回退。
    # 仍待处理：canonical 台词产出后，这里应换成从 290 条冻结台词里
    # 逐 base 挑选 + 人工审的真台词，本次只是先把「教模型说禁句」这个矛盾消掉。
    return "（示范·陈述句，仅参考语气句式、别照抄内容）比如：“单子上写的时间是下个月十号。”"


# T3 关键轮忌讳的结论式表述：供 _filter_examples_for_level 过滤 few-shot + StanceMarkers 度量用。
# 内部方案 后 prompt 侧不再枚举禁用词清单（压成原则句），本 tuple 成为词表防线的唯一载体
# （代码侧， 不动 tuple 本身）；原「前 8 个作示例词、新增追加到尾部」的顺序约束随之解除。
# K18d 追加让步/顺从类语气词：K18c 的 HOT-NEG-44-T3 用“行吧…”开头把配合立场漏了出去
# （探针 4 分），而它既非情绪词也非否定结论，原清单一个都不命中 —— StanceMarkers 作为
# 结论分的客观对照，漏掉这一类会让 T3 的程序化读数虚低。
_T3_STANCE_MARKERS = (
    "不需要", "不用了", "不要再", "不打算", "不接受", "没法接受", "等不起",
    "太久", "别打", "别再", "处理不了", "拿不出", "没法配合", "不用你们管",
    "行吧", "算了", "好吧", "那行", "无所谓", "随便吧",
)


def _pre_key_stance_ban_v7(level: str) -> str:
    """v7 非关键轮的提前摊牌禁令（K18b 复跑第三处修正 R3）。

    修正的偏离：关键轮子句已给出具体禁用句式清单，非关键轮却只有抽象表述
    “不提前摊出来”；而数据侧 mission 要点本身多写成结论口气（如“要求对方停止拨打”
    “不想继续听”），模型照着念就在关键轮之前把同向结论先说了
    （K18b MAR-ALL-23-T3 关键轮在 T5，T4 已说“以后别再打了”），
    关键轮信息量被提前消耗。此处把禁令对称到非关键轮：只要求“往后放”（延后到
    关键轮说）而不是“不许说”，因此不影响任务收口；T1 不加（T1 本就要求显式态度，
    额外禁令会与“把情绪说出来”相互干扰）。

    内部方案：删掉枚举词表（原 _T3_STANCE_MARKERS[:8] 的逐个列举）——改动5 明确
    常驻位置不放词表（例词清单会被当成逐轮推荐模板，内部方案 实测填充词起头率 24%→92%），
    压成一句原则。词表防线仍在代码侧生效（_T3_STANCE_MARKERS 供 _filter_examples_for_level
    过滤 few-shot + StanceMarkers 度量），与 prompt 侧解耦（不动 tuple 本身）。
    """
    if level == "T1":
        return ""
    return "（写成表态口气的要点，那轮之前先按住）"


def _filter_examples_for_level(examples: list[str], level: str) -> list[str]:
    """T3 关键轮要求结论不上文本，few-shot 里的结论式真实语料会被照抄（纸面校验发现）。

    仅 T3 过滤；过滤后不足 2 条则保留原样（示范缺失比风格偏差更伤自然度）。
    """
    if level != "T3" or not examples:
        return examples
    kept = [e for e in examples if not any(m in e for m in _T3_STANCE_MARKERS)]
    return kept if len(kept) >= 2 else examples


def _transparency_clause(level: str, t1_words: list[str]) -> str:
    """透明度约束：非关键轮统一克制中性；关键轮按 T1/T2/T3 分级。"""
    return "非关键轮：克制中性，禁止明确情绪词。轮到【关键事件】时：" + _transparency_key_clause(level, t1_words)


def _render_forbidden(forbidden: list[Any]) -> str:
    """渲染硬约束列表（数据文案同步当前口径，避免 prompt 自相矛盾）。"""
    items = []
    for item in forbidden or []:
        s = str(item)
        s = re.sub(r"不超过\s*30\s*字", f"不超过{MAX_UTTERANCE_CHARS}字", s)
        s = re.sub(r"句子要说完整[，,]?不得中途截断", "句子基本说完整，允许半句收尾", s)
        if s.strip():
            items.append(f"- {s}")
    return "\n".join(items) if items else "（无）"


# ---- few-shot 示范轮（角色×动态；示范句均避开明确情绪词）----
_FEWSHOT_EXAMPLES: dict[tuple[str, str], list[str]] = {
    ("collection", "all_positive"): [
        "嗯，这个安排可以，我周五之前把钱还上。",
        "行，就按你说的办，我记下来了。",
        "好的，那麻烦你帮我登记一下。",
    ],
    ("collection", "all_negative"): [
        "我现在手头真紧，拿不出这笔钱。",
        "你们不要再打来了，我真的处理不了。",
        "这事我自有打算，不用你们管。",
    ],
    ("collection", "pos_to_neg"): [
        "本来想好好配合的，这么催我接受不了。",
        "说好的方案又变，这还怎么谈？",
    ],
    ("collection", "neg_to_pos"): [
        "要是能分两期，我倒是可以试试。",
        "行吧，那就按这个方案先走着。",
    ],
    ("marketing", "all_positive"): [
        "听着还行，你把重点再给我说说。",
        "可以啊，这个活动我可以了解一下。",
        "嗯，那你把办理方式说一下。",
    ],
    ("marketing", "all_negative"): [
        "我不需要，以后不要再打来了。",
        "我在忙，先这样吧。",
        "说了不要了，你们怎么还打。",
    ],
    ("marketing", "pos_to_neg"): [
        "刚才还好好的，怎么突然又要收费了？",
        "你这么说我就没法继续听了。",
    ],
    ("marketing", "neg_to_pos"): [
        "要真是免费的，那你说来听听。",
        "嗯，那我先听两句，看划不划算。",
    ],
    ("hotline", "all_positive"): [
        "好的，麻烦你帮我查一下订单。",
        "嗯，那就这样处理，我等你消息。",
        "行，按你说的步骤来。",
    ],
    ("hotline", "all_negative"): [
        "等了好久还没处理，到底卡在哪儿了？",
        "这个处理结果我没法接受。",
        "我都说了好几遍了，怎么还要重新来？",
    ],
    ("hotline", "pos_to_neg"): [
        "刚才还答应得好好的，怎么又让我等？",
        "说好的处理时间又变了？",
    ],
    ("hotline", "neg_to_pos"): [
        "要是今天能处理好，我可以等。",
        "行，那就照你说的办。",
    ],
    ("", "all_positive"): ["嗯，好的，我知道了。", "行，那就按你说的来。"],
    ("", "all_negative"): ["我现在不想谈这个。", "先这样吧，我还有事。"],
    ("", "pos_to_neg"): ["你这么说，我就没法配合了。", "怎么又变卦了？"],
    ("", "neg_to_pos"): ["那你说说看，怎么个办法。", "行，我听听你的方案。"],
}


_ROLE_REALISM = {
    "collection": (
        "被催收的人最常做四件事，你可以按处境挑着做，不必都做：探对方是谁、凭什么找你；"
        "对账单里说不通的地方提出疑问；说自己眼下的难处；就金额、期限或方式讨个条件。"
    ),
    "hotline": (
        "主动打客服的人最常做的是：把自己遇到的问题讲清楚，对不合自己记录的说法提出异议，"
        "追问到什么时候能有结果。你是来解决问题的，不是来谈条件的。"
    ),
    "marketing": (
        "接到推销电话的人最常做的是：先探对方是哪家、怎么拿到自己号码的；"
        "对说不清的地方将信将疑；不想听就找个由头把电话往后推。"
    ),
}


def _realism_block(role: str, role_type: str, dynamics: str) -> str:
    """【真人打电话的习惯】：把真实通话里的用户行为写成可执行的倾向。

    数据依据是 `（未发布）` 里 524 通真实通话（催收 323 / 客服 182 / 营销 19，
    仅取双轨可强标记区分的通话）。各行为的按通占比：

    | 行为 | 催收 | 客服 | 营销 |
    |---|---|---|---|
    | 试探对方身份与来意 | 38.4% | 18.1% | 26.3% |
    | 讨价与条件 | 31.9% | 0.5% | 0.0% |
    | 诉困难 | 29.7% | 7.1% | 10.5% |
    | 否认或不认账 | 20.4% | 23.1% | 26.3% |
    | 拖延与回避 | 11.1% | 12.1% | 10.5% |

    合成对话与真人的差距正在这里：同一套指标下，`n7_20260826_qa3plus` 的
    诉困难只有 2.8%、否认 1.9%、答应或配合 0.9%（真人 44~69%），
    而抱怨被骚扰 11.1% 反而高于真人（1.6~5.6%）—— 演出来的用户比真人更爱抗议、
    更不爱配合，也更不会讲自己的难处。

    **不给例句，只给行为。** 的教训：本块常驻 system prompt、每轮都在模型视野里，
    上一次给出「嗯/那个/这个」这样的例词清单，结果被当成推荐模板 ——
    填充词起头率 24%→92%，「嗯」成了逐轮口癖。所以这里一律写"做什么"，不写"怎么说"。

    **只约束非关键轮。** 关键事件那一轮以【语气提示】为准。否则「说自己的难处」
    会在 T3 关键轮里变成一句带线索的铺垫，把 T3 的无线索设计（B−C 的承重结构）拆掉。
    """
    lines = ["【真人打电话的习惯】（这些是平时几轮的倾向，"
             "轮到【关键事件】那轮一律以上面的【语气提示】为准）"]
    if role_type in OUTBOUND_ROLE_TYPES:
        lines.append(
            "· 这通电话是对方打来的：你接起时并不知道对方是谁、代表哪家、为什么打来。"
            "凡属对方那边的事（这通电话为哪笔账、有哪些档位价格、活动包含什么），"
            "都等客服亲口说了你才知道，之前绝不主动说出口 —— 哪怕【要办的事】里写着它。"
        )
    lines.append(
        "· 你不会主动说对自己不利的话：对方没问到、说出来只会让自己更被动的信息，"
        "不主动交代；被直接问到才答，也可以先答得含糊些。"
        "但【要办的事】里明确要你交代的，照做。"
    )
    role_line = _ROLE_REALISM.get(role)
    if role_line:
        lines.append(f"· {role_line}")
    lines.append(
        "· 真人说话长短不齐：多数轮十几二十个字，遇到只是应一声、认个账、"
        "答应个时间的轮次，几个字就够，不用每轮都凑成一句完整交代。"
    )
    return "\n".join(lines) + "\n"


def _fewshot_examples(role: str, dynamics: str) -> list[str]:
    # 内部方案/：优先真实语料风格池（（未发布），
    # 脱敏真实用户句，style_stats 同源）；文件缺失/无对应角色时回退内置示例。
    pool = _load_style_pool()
    if pool:
        hits = [p["text"] for p in pool if p.get("role") == role]
        if hits:
            return hits[:5]
    return _FEWSHOT_EXAMPLES.get((role, dynamics)) or _FEWSHOT_EXAMPLES.get(("", dynamics), [])


_STYLE_POOL_CACHE: list[dict] | None = None


def _load_style_pool() -> list[dict]:
    """风格池懒加载（只读一次；文件不存在返回空，优雅降级）。"""
    global _STYLE_POOL_CACHE
    if _STYLE_POOL_CACHE is not None:
        return _STYLE_POOL_CACHE
    path = Path(__file__).resolve().parents[3] / "（未发布）"
    pool: list[dict] = []
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    pool.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            pool = []
    _STYLE_POOL_CACHE = pool
    return pool


def _render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "（无）"
    lines = []
    for m in history:
        who = "用户" if m["role"] == "user" else "客服"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


def _render_history_compressed(history: list[dict[str, str]], keep_msgs: int = 8,
                               summary_chars: int = 24) -> str:
    """内部方案（v7）：历史压缩——近 keep_msgs 条（约 4 轮）全文保留，
    更早的每条截取前 summary_chars 字拼成一句摘要，防长对话 prompt 膨胀。"""
    if not history:
        return "（无）"
    recent = history[-keep_msgs:]
    older = history[:-keep_msgs]
    lines: list[str] = []
    if older:
        parts = []
        for m in older:
            who = "用户" if m["role"] == "user" else "客服"
            c = (m.get("content") or "").strip()
            parts.append(f"{who}: {c[:summary_chars]}{'…' if len(c) > summary_chars else ''}")
        lines.append("【更早对话摘要】" + " ".join(parts))
    for m in recent:
        who = "用户" if m["role"] == "user" else "客服"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


# ---- 透明度层级归一化 ----
_LEVEL_BY_NAME = {
    "T1": "T1", "explicit": "T1",
    "T2": "T2", "implicit_consistent": "T2",
    "T3": "T3", "prosody_only": "T3",
}


def normalize_leakage_level(value: str) -> str:
    """把 leakage_label/leakage_level 归一为 T1/T2/T3；无法识别返回 ""。"""
    return _LEVEL_BY_NAME.get(str(value or "").strip(), "")


# ---- base_scenario_id 推导（内部方案：instruction 表索引键不含透明度尾缀）----
_LEAKAGE_SUFFIX_RE = re.compile(r"-T[123]$")


def _base_scenario_id_of(scenario_id: str) -> str:
    """剥掉 scenario_id 末尾的 -T1/-T2/-T3，得到 instruction 表的索引键。

    三档共用一条 instruction 是「韵律三级拉平」在数据结构上的落点：索引键里没有透明度，
    某一档就没有偷偷换韵律的可能，不靠代码约定维持。
    """
    return _LEAKAGE_SUFFIX_RE.sub("", str(scenario_id or "").strip())


# tts_control 回落角色设定短语的身份词
_ROLE_IDENTITY_WORD = {
    "collection": "欠款客户",
    "hotline": "来电客户",
    "marketing": "意向客户",
}


def _instruct_from_control(state: str, tts_control: str, *, scenario_id: str = "", turn: int = 0) -> str:
    """生成情感 instruct 描述（保持「用{tts_control}的语气」包装 + ≤80 字符硬上限）。

    内部方案：tts_control 改为查预生成表所得，格式违约不再是「LLM 偶尔跑偏」而是
    表数据坏了或接线错了，因此原 warning 升为 raise——静默降级会把「音频韵律与 state
    无关」这类问题一路藏到听测阶段。
    """
    if len(tts_control) > MAX_TTS_CONTROL_CHARS:
        raise ValueError(
            f"tts_control 超过 {MAX_TTS_CONTROL_CHARS} 字符硬上限"
            f"（实际 {len(tts_control)}）：scenario={scenario_id or '（未传入）'} turn={turn} "
            f"state={state} tts_control={tts_control!r}"
        )
    has_prosody_kw = any(w in tts_control for w in RATE_WORDS) or any(w in tts_control for w in VOLUME_WORDS)
    if not tts_control or not has_prosody_kw:
        raise ValueError(
            f"tts_control 不符契约（须非空且含语速/音量词）："
            f"scenario={scenario_id or '（未传入）'} turn={turn} "
            f"state={state} tts_control={tts_control!r}"
        )
    return f"用{tts_control}的语气"


def _prosody_from_control(tts_control: str) -> dict[str, Any]:
    """从 tts_control 文本提取韵律参数（关键词启发式）。

    内部方案 修订：新增「音量压低」→0.8（hushed 态豁免「音量≥正常」契约）。

    2026-08-26 字段审计：本函数产出当前**没有后端消费**——TTSGateway 的 dashscope
    路径不下发 prosody，部分后端只读 `speed_ratio` 键（与此处 rate/volume 键名
    不匹配）。韵律实际由 instruct 文本中的「语速…／音量…」描述承载。保留本函数是
    为了不动历史 run 的音频口径（接线会真实改变合成结果、使历史结果不可比），
    是否接线需单独评估。
    """
    prosody: dict[str, Any] = {}
    for word, rate in (("语速偏快", 1.1), ("语速快", 1.2), ("语速中等", 1.0)):
        if word in tts_control:
            prosody["rate"] = rate
            break
    for word, volume in (("音量略升", 1.15), ("音量大", 1.25), ("音量压低", 0.8), ("音量正常", 1.0)):
        if word in tts_control:
            prosody["volume"] = volume
            break
    return prosody
