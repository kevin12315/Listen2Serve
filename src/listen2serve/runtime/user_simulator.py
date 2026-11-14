"""骨架驱动的多轮用户模拟器。

场景提供身份、目标、私有事实、剧情要点、关键事件和轮次预算。每轮生成紧凑 JSON，
运行时校验角色、状态、字数与词表；首次关键轮锁存，接近硬上限时强制关键事件。
声音指令与标签从场景表读取；干净文本与合成文本分别保留。"""

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
from listen2serve.evaluation.closing_check import _USER_CLOSING
from listen2serve.gateway.llm_gateway import LLMGateway
from listen2serve.gateway.tts_gateway import TTSGateway
from listen2serve.runtime.repayment_facts import unsupported_repayment_completion
from listen2serve.runtime.tts_instructions import load_table as load_instruction_table
from listen2serve.runtime.tts_instructions import validate_entry as validate_instruction_entry
from listen2serve.runtime.tts_tags import allowed_tags_from_entry, apply_tag_policy, find_tags

logger = logging.getLogger(__name__)

MIN_UTTERANCE_CHARS = 10  # 统一字数下限（含标点；不再分关键/非关键轮口径）
MAX_UTTERANCE_CHARS = 60  # 统一字数上限（含标点）
MAX_TTS_CONTROL_CHARS = 80  # tts_control 长度硬上限（「用{…}的语气」包装契约沿用）

# Only these existing source templates have deterministic prerequisite checks.
_FINANCE_TRIGGER_RULES = {
    "客服说明拖欠影响并试图促成当前还款安排后，才明确摊牌拒绝": ("影响说明", "还款协商"),
    "客服确认官方还款渠道并引导协商还款安排后，用户摊牌表示愿意配合确认具体还款方式和期限": (
        "官方渠道",
        "还款协商",
    ),
    "客服说明继续拖欠的影响并提出可操作的处理方案后，用户开始回应处理意愿并询问还款方式与期限。": (
        "影响说明",
        "可操作方案",
    ),
    "客服给出可行或灵活的还款方案后，用户才放下抗拒并摊牌配合。": ("可操作方案",),
}

STAGE_DIRECTION_RE = re.compile(r"[（(][^（）()]*[)）]")

STATES = ("neutral", "cooperative", "displeased", "urgent", "doubtful", "hushed")

OUTBOUND_ROLE_TYPES = ("outbound_pressure", "outbound_negotiation")
OUTBOUND_OPENING_POOL = (
    "喂，你好。",
    "喂？",
    "喂，你好？",
    "哎，你好。",
    "你好。",
    "喂，哪位？",
    "你好，哪位？",
    "嗯，你好。",
)

RATE_WORDS = ("语速中等", "语速偏快", "语速快", "语速很快")
VOLUME_WORDS = ("音量正常", "音量略升", "音量大", "音量压低")


@dataclass
class UserTurnResult:
    """用户模拟器一轮输出。"""

    text: str = ""
    text_tts: str = ""
    pcm16: bytes = b""
    state: str = ""
    tts_control: str = ""
    is_key_turn: bool = False
    key_form: str = ""
    covered: list[int] = field(default_factory=list)
    forced_key_turn: bool = False
    end_call: bool = False
    interrupt: bool = False  # 本轮以打断方式开口（仅关键轮合法）
    state_mismatch: bool = False  # 关键轮 state 与数据不符被覆盖
    vocab_violation: bool = False  # 词表防线重试后仍违规（不阻断，仅标记）
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
        self.prompt_log_path = Path(prompt_log_path) if prompt_log_path else None
        self.role_type = role_type
        if prosody_arm not in ("state", "neutral"):
            raise ValueError(f"prosody_arm 非法: {prosody_arm!r}（只接 state / neutral）")
        self.prosody_arm = prosody_arm

        self.sim_thinking = sim_thinking

        raw_mission = user_script.get("mission") or []
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
            raise ValueError("user_script 缺 mission/key_event")
        key_state = str(self.key_event.get("state") or "")
        if key_state not in STATES:
            raise ValueError(f"key_event.state 非法: {key_state!r}")

        self.base_scenario_id = base_scenario_id or _base_scenario_id_of(scenario_id)
        entry = (
            dict(instruction_entry)
            if instruction_entry is not None
            else load_instruction_table(instruction_table_path).entry(self.base_scenario_id)
        )
        validate_instruction_entry(self.base_scenario_id, entry)
        if entry["key_state"] != key_state:
            raise ValueError(
                f"instruction 表 key_state={entry['key_state']!r} 与 key_event.state={key_state!r} "
                f"不一致：base={self.base_scenario_id}"
            )
        self._instruction_entry = entry
        self.candidate_states: tuple[str, ...] = tuple(entry["candidate_states"])
        self._allowed_tags = allowed_tags_from_entry(entry)
        self._t1_sample_words = self._sample_state_words(key_state)

        self._turn_count = 0
        self._key_turn_no: int | None = None  # 锁存的唯一关键轮号
        self._covered: set[int] = set()
        self._ended = False
        self._last_end_call_rejected = False
        self.termination_reason = ""  # user_closed |（硬上限由 orchestrator 标 max_turns）
        self._history: list[dict[str, str]] = []
        self._last_state = "neutral"  # 上一轮实际 state（_state_for_turn 算预期值用）
        self._draft_done = False
        self._draft_interrupt = False
        self.stats = {
            "llm_calls": 0,
            "json_failures": 0,
            "retries": 0,
            "truncation_retries": 0,
            "forced_key_turn": 0,
            "state_mismatch": 0,
            "vocab_violation": 0,
            "end_call_rejected": 0,
            "key_downgrade": 0,
            "outbound_opening": 0,
            "end_call_no_cue": 0,
            "end_call_cue_rejected": 0,
            "fact_violations": 0,
            "state_drift": 0,
            "tag_stripped": 0,
            "stage_direction_stripped": 0,
        }

    def _sample_state_words(self, state: str) -> list[str]:
        """从关键状态词表确定性抽样，供显式条件的关键轮提示和校验使用。"""
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
        """按基础场景号确定关键轮句式，供结果分组使用。"""
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

    def next_turn(
        self, agent_reply: str = "", tool_calls: list[dict[str, Any]] | None = None
    ) -> UserTurnResult | None:
        """生成下一轮用户发言（文本 + 音频）；end_call 轮之后返回 None。"""
        if self._ended:
            return None
        self._draft_done = False  # 新客服轮：打断草案缓存失效
        n = self._turn_count + 1
        if n == 1 and not agent_reply and self.role_type in OUTBOUND_ROLE_TYPES:
            return self._outbound_opening_turn(n)
        forced = (not self.key_completed) and n >= self.hard_budget - 2

        parsed = self._elicit(n, agent_reply, forced=forced)

        text = parsed["text"]
        state = self._state_for_turn(parsed)
        is_key = parsed["is_key_turn"]
        end_call = parsed["end_call"]
        covered_new = parsed["covered"]

        tts_control = self._assemble_instruct(state, is_key)
        tag_raw = find_tags(text)  # LLM 自行输出的标记（协议已删该字段，属违约）先留档
        forced_tag, tag_res = self._tagged_text(text, state, is_key)
        text = tag_res.text
        pcm16 = self._synthesize(tag_res.text_tts, state, tts_control)

        if is_key:
            self._key_turn_no = n
        self._covered.update(covered_new)
        self._turn_count = n
        self._last_state = state
        self._last_end_call_rejected = bool(parsed.get("_end_call_rejected"))
        if end_call:
            self._ended = True
            self.termination_reason = "user_closed"
            if not any(w in text for w in _USER_CLOSING):
                self.stats["end_call_no_cue"] += 1

        if agent_reply:
            self._history.append({"role": "assistant", "content": agent_reply})
        self._history.append({"role": "user", "content": text})

        return UserTurnResult(
            text=text,
            text_tts=tag_res.text_tts,
            pcm16=pcm16,
            state=state,
            tts_control=tts_control,
            is_key_turn=is_key,
            covered=covered_new,
            key_form=self.key_form if is_key else "",
            forced_key_turn=parsed.get("_forced", False),
            end_call=end_call,
            interrupt=parsed.get("interrupt", False),
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

    def _outbound_opening_turn(self, n: int) -> UserTurnResult:
        """外呼首轮从被动接听应答池确定性选句，剧情从后续轮次推进。"""
        idx = zlib.crc32(self.scenario_id.encode("utf-8")) % len(OUTBOUND_OPENING_POOL)
        text = OUTBOUND_OPENING_POOL[idx]
        state = "neutral"
        tts_control = self._assemble_instruct(state, is_key=False)
        tts_err: Exception | None = None
        for k in range(len(OUTBOUND_OPENING_POOL)):
            cand = OUTBOUND_OPENING_POOL[(idx + k) % len(OUTBOUND_OPENING_POOL)]
            try:
                pcm16 = self._synthesize(cand, state, tts_control)
                text = cand
                tts_err = None
                break
            except Exception as exc:  # noqa: BLE001 - 换句重试；耗尽后抛原错
                tts_err = exc
                logger.warning(
                    "开场 TTS 失败换句重试: scenario=%s text=%r err=%s", self.scenario_id, cand, str(exc)[:80]
                )
        if tts_err is not None:
            raise RuntimeError(f"outbound 开场 TTS 全池失败: scenario={self.scenario_id} last={tts_err}")
        self._turn_count = n
        self.stats["outbound_opening"] += 1
        self._history.append({"role": "user", "content": text})
        if self.prompt_log_path is not None:
            try:
                record = {
                    "turn": n,
                    "outbound_opening": True,
                    "forced": False,
                    "retried": False,
                    "llm_params": None,
                    "messages": [{"role": "user", "content": text}],
                }
                self.prompt_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.prompt_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                logger.warning("user_prompts 落盘失败（降级跳过）：scenario=%s err=%s", self.scenario_id, exc)
        return UserTurnResult(
            text=text,
            text_tts=text,
            pcm16=pcm16,
            state=state,
            tts_control=tts_control,
            is_key_turn=False,
            covered=[],
            forced_key_turn=False,
            end_call=False,
            interrupt=False,
            state_mismatch=False,
            vocab_violation=False,
            llm_debug={"outbound_opening": True},
        )

    def peek_interrupt(self, agent_reply: str = "", tool_calls: list[dict[str, Any]] | None = None) -> bool:
        """关键事件未完成且 key_event.interrupt=true 时，预生成下一轮草案取
        interrupt 标记；每客服轮至多预生成一次（部分转写 ≥8 字后）。"""
        if not self.key_event.get("interrupt") or self.key_completed or self._ended:
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
        except Exception as exc:  # noqa: BLE001 - 预生成失败不打断（保守）
            logger.warning("interrupt 预生成失败: %s", exc)
            self._draft_interrupt = False
        self._draft_done = True
        return self._draft_interrupt

    def _elicit(self, n: int, agent_reply: str, *, forced: bool, dry_run: bool = False) -> dict[str, Any]:
        """一轮协议生成：首次调用 → 汇总硬违规 → 带反馈重试一次 → 残余按策略处置。

        硬失败（两次后仍 JSON 不可解析/空文本/复读客服/字数越界/state 非法）→
        抛错，由批跑层任务级重试（评测通路无兜底原则不变）。
        """
        messages = self._build_messages(n, agent_reply, forced=forced)
        parsed, problems = self._call_and_validate(messages, n, agent_reply, seed_offset=0)
        if parsed is not None:
            problems.extend(self._finance_trigger_violations(parsed, agent_reply, forced=forced))
            first_is_key = bool(parsed.get("is_key_turn")) and not self.key_completed
            first_hits = self._vocab_defense(parsed["text"], forced or first_is_key)
            if first_hits:
                if self.leakage_label == "T3" and (forced or first_is_key) and self.key_event.get("surface"):
                    problems.append(
                        f"中性关键轮出现显式状态或立场自述（命中{first_hits}）。"
                        f"【中性关键轮重写】已有表面业务话题：{self.key_event['surface']}。"
                        "删除状态和立场自述、态度结论，只围绕该话题作中性业务核对或询问；"
                        "自然承接客服上文，不要求照抄固定句子。真实态度保留在JSON state与声音中，不写进text"
                    )
                else:
                    problems.append(
                        f"台词命中词表防线（{first_hits}）：本轮禁止任何明确情绪词"
                        "（T1 关键轮须改用要求状态的明确情绪词），请改写后重新输出"
                    )
        if problems and not dry_run:
            self.stats["retries"] += 1
        if problems:
            retry_note = (
                "你刚才的输出不合格："
                + "；".join(problems)
                + "。这是同一用户轮的修订，客服上文没有变化，不是下一轮；"
                "只修正所列问题，不得借修文跳到下一剧情或提前触发关键事件。请重新输出合规的紧凑 JSON。"
            )
            retry_draft = json.dumps(parsed, ensure_ascii=False) if parsed is not None else "(上次输出不合格)"
            retry_messages = messages + [
                {"role": "assistant", "content": retry_draft},
                {"role": "user", "content": retry_note},
            ]
            parsed2, _problems2 = self._call_and_validate(retry_messages, n, agent_reply, seed_offset=500)
            if parsed2 is not None:
                parsed = parsed2
            if parsed is None:
                raise RuntimeError(f"用户模拟器协议两次失败：scenario={self.scenario_id} turn={n}")
            hard_left = self._hard_violations(parsed, n, agent_reply)
            hard_left.extend(self._finance_trigger_violations(parsed, agent_reply, forced=forced))
            if hard_left:
                raise RuntimeError(
                    f"用户模拟器协议重试后仍硬违规：scenario={self.scenario_id} turn={n} problems={hard_left}"
                )

        if parsed is None:
            raise RuntimeError(f"用户模拟器协议两次失败：scenario={self.scenario_id} turn={n}")

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
            is_key = False
            parsed["is_key_turn"] = False
            parsed.pop("_forced", None)
            if not dry_run:
                self.stats["key_downgrade"] += 1
            logger.warning("is_key_turn 重复声明被降级：scenario=%s turn=%s", self.scenario_id, n)

        if is_key and parsed["state"] != self.key_event.get("state"):
            parsed["_state_mismatch"] = True
            parsed["state"] = str(self.key_event.get("state"))
            if not dry_run:
                self.stats["state_mismatch"] += 1

        end_call = bool(parsed.get("end_call"))
        key_done_after = self.key_completed or is_key
        if end_call and not key_done_after:
            parsed["end_call"] = False
            parsed["_end_call_rejected"] = True
            end_call = False
            if not dry_run:
                self.stats["end_call_rejected"] += 1

        if end_call and not self._can_end_call(parsed):
            parsed["end_call"] = False
            parsed["_end_call_rejected"] = True
            if not dry_run:
                self.stats["end_call_rejected"] += 1
                self.stats["end_call_cue_rejected"] += 1

        parsed["interrupt"] = bool(parsed.get("interrupt") and is_key and self.key_event.get("interrupt"))

        hits = self._vocab_defense(parsed["text"], is_key)
        if hits is not None:
            parsed["_vocab_violation"] = True
            if not dry_run:
                self.stats["vocab_violation"] += 1
            logger.warning(
                "vocab violation（重试后仍命中，保留输出）：scenario=%s turn=%s hits=%s",
                self.scenario_id,
                n,
                hits,
            )
        if not dry_run and self.prompt_log_path is not None:
            self._log_elicit(n, messages, retried=bool(problems), forced=forced)
        return parsed

    def _finance_trigger_violations(
        self, parsed: dict[str, Any], agent_reply: str, *, forced: bool
    ) -> list[str]:
        """Check explicitly supported source after conditions, not a draft's key flag.

        Forced-budget handling and unsupported free-form hints retain their
        existing behavior. Source/user amounts are comparison values, not proof
        that an assistant action happened.
        """
        if (
            forced
            or self.key_completed
            or not parsed.get("is_key_turn")
            or self.role not in ("", "collection")
        ):
            return []
        rule = _FINANCE_TRIGGER_RULES.get(str(self.key_event.get("trigger_hint") or "").strip())
        if rule is None:
            return []
        spoken = [str(row.get("content") or "") for row in self._history if row.get("role") == "assistant"]
        if agent_reply:
            spoken.append(agent_reply)
        evidence = _finance_prerequisite_evidence(spoken)
        if "可操作方案" in rule:
            evidence["可操作方案"] = [text for text in spoken if _finance_executable_plan(text)]
        missing = [name for name in rule if not evidence[name]]
        when = str(self.disclosure_plan.get("debt_amount") or "")
        debt = (self.user_script.get("user_facts") or {}).get("debt_amount")
        amount = _finance_number_tokens(str(debt or ""))
        if "客服核对欠款数额时才说" in when and len(amount) == 1:
            expected = amount[0]
            amount_spoken = spoken
            if rule == ("可操作方案",):
                # This source template defines its debt amount as the current period's amount.
                amount_spoken = [
                    re.sub(r"本期(?:应还(?:款)?(?:金额|数额)?|还款金额|金额)", "欠款总额", text)
                    for text in spoken
                ]
            if expected in _finance_number_tokens(str(parsed.get("text") or "")) and not any(
                _finance_amount_check(text) for text in amount_spoken
            ):
                missing.append("客服核对该欠款金额")
        if not missing:
            return []
        return [
            "源关键事件前置条件尚缺可识别的客服发言证据："
            + "、".join(missing)
            + "。此轮先自然追问或回应现有业务，is_key_turn=false；待客服实际完成原前置条件后再进入关键轮"
        ]

    def _log_elicit(self, n: int, messages: list[dict[str, str]], *, retried: bool, forced: bool) -> None:
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
                os.fsync(f.fileno())  # NAS 写入时序防护（与 PartialStore 同口径）
        except OSError as exc:
            logger.warning(
                "user_prompts 落盘失败（降级跳过）：scenario=%s turn=%s err=%s", self.scenario_id, n, exc
            )

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
                f"state 非法: {parsed.get('state')!r}（本场景候选集 {'/'.join(self.candidate_states)}）"
            )
        problems.extend(self._fact_violations(text, agent_reply))
        return problems

    def _call_and_validate(
        self,
        messages: list[dict[str, str]],
        n: int,
        agent_reply: str,
        seed_offset: int,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """调用用户模型，解析 JSON 并校验台词、状态、关键事件及终止条件。

        非推理后端使用 320 token，上限截断时以 1024 重试一次；Gemini 使用 2048。"""
        model_name = (self.llm_model or "").lower().split("/")[-1]
        reasoning = model_name.startswith("gemini")
        limit = 2048 if reasoning else 320
        result = self.llm.chat(
            messages,
            model=self.llm_model,
            temperature=0.7,
            max_tokens=limit,
            seed=self.seed * 1000 + n + seed_offset,
            thinking=self.sim_thinking,
        )
        self.stats["llm_calls"] += 1
        if result.finish_reason == "length" and not reasoning:
            self.stats["truncation_retries"] += 1
            result = self.llm.chat(
                messages,
                model=self.llm_model,
                temperature=0.7,
                max_tokens=1024,
                seed=self.seed * 1000 + n + seed_offset + 77,
                thinking=self.sim_thinking,
            )
            self.stats["llm_calls"] += 1
        parsed = _parse_protocol(result.text)
        if parsed is None:
            self.stats["json_failures"] += 1
            return None, ["输出不是合规 JSON 协议"]
        text = str(parsed.get("text") or "").strip()
        _n_sd = len(STAGE_DIRECTION_RE.findall(text))
        if _n_sd:
            self.stats["stage_direction_stripped"] += _n_sd
            text = STAGE_DIRECTION_RE.sub("", text).strip()
        parsed["text"] = text
        problems: list[str] = []
        if not text:
            return None, ["text 为空"]
        fact_problems = self._fact_violations(text, agent_reply)
        if fact_problems:
            self.stats["fact_violations"] += 1
            problems.extend(fact_problems)
        if not (MIN_UTTERANCE_CHARS <= len(text) <= MAX_UTTERANCE_CHARS):
            problems.append(
                f"text 需 {MIN_UTTERANCE_CHARS}–{MAX_UTTERANCE_CHARS} 字（含标点），当前 {len(text)} 字"
            )
        if agent_reply and (text in agent_reply or agent_reply[:15] in text):
            problems.append("text 与客服上轮回复高度重合，只说用户自己的话")
        if parsed.get("state") not in self.candidate_states:
            problems.append(f"state 必须是 {'/'.join(self.candidate_states)} 之一")
        is_key = bool(parsed.get("is_key_turn"))
        if is_key and not self.key_completed and parsed.get("state") != self.key_event.get("state"):
            problems.append(f"关键轮 state 必须是 {self.key_event.get('state')}（关键事件指定状态）")
        end_call = bool(parsed.get("end_call"))
        if end_call and not (self.key_completed or is_key):
            problems.append("还不能挂断，你还有事没办（关键事件完成后才可道别收尾）")
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

    def _initial_disclosure_locks(self, agent_reply: str = "") -> dict[str, Any]:
        """Only the first user turn with no assistant speech has a certain gate.

        A later reply does not prove a compound disclosure condition was met;
        those existing conditions remain in the prompt unchanged.
        """
        if (
            self._turn_count
            or agent_reply.strip()
            or any(
                item.get("role") == "assistant" and str(item.get("content") or "").strip()
                for item in self._history
            )
        ):
            return {}
        facts = self.user_script.get("user_facts") or {}
        return {
            key: facts[key]
            for key, when in self.disclosure_plan.items()
            if key in facts
            and str(facts[key]).strip()
            and re.search(r"客服[^。；\n]*(?:询问|追问|问及|问到|核实|核对)", str(when))
            and re.search(r"才|后|时", str(when))
        }

    def _fact_violations(self, text: str, agent_reply: str = "") -> list[str]:
        """Reject narrow fact contradictions and unasked first-turn disclosure.

        This does not infer arbitrary facts or resolve ambiguous relative dates.
        Other source facts authorizing a month-end date disable that date check.
        """

        # Match only a source-derived literal core, not arbitrary paraphrases.
        # The first clause after “发现” retains the concrete issue while dropping
        # introductory timing. Optional “有” covers its observed spoken variant.
        def normalize(value: str) -> str:
            return re.sub(r"[\W_]", "", value).replace("有", "")

        public_topics = [self.user_script.get(name, "") for name in ("identity", "goal", "user_goal")]
        public_topics += self.mission + [self.key_event.get("surface", "")]
        for key, value in self._initial_disclosure_locks(agent_reply).items():
            core = re.split(r"[，,。；;\n]", _fact_text(key, value).split("发现", 1)[-1], maxsplit=1)[0]
            marker = normalize(core)
            if (
                marker
                and marker in normalize(text)
                and not any(marker in normalize(str(topic)) for topic in public_topics)
            ):
                return [f"{key}首轮披露时机未到：尚无客服询问；只说场景允许的一般诉求，暂不透露该私有细节"]
        address_problems = self._unsupported_address_claims(text)
        if address_problems:
            return address_problems
        facts = self.user_script.get("user_facts") or {}
        if self.role in ("", "collection"):
            repayment_problems = unsupported_repayment_completion(text, facts, self.mission, self.db_seed)
            if repayment_problems:
                return repayment_problems
        note = str(facts.get("income_note") or "")
        received = re.search(r"(?:工资|薪水|薪资)(?:刚刚?|已经|已)(?:到账|到手|发了|发下来)", note)
        source_text = " ".join(str(value) for value in facts.values())
        authorized_wait = re.search(
            r"(?:等(?:到)?[^，。！？；]{0,8}(?:发(?:放)?工资|工资(?:到账|到手|发放))"
            r"|(?:下次|下月|下个月|月底)[^，。！？；]{0,8}(?:发薪|发工资|工资到账))",
            source_text,
        )
        clauses = re.split(r"[，,。；;！!\n]", text)
        for clause in clauses:
            if any(mark in clause for mark in ("？", "?", "如果", "假如", "要是")):
                continue
            if received and not authorized_wait:
                waiting = re.search(
                    r"(?:等(?:到)?[^，。！？；]{0,8}(?:发(?:放)?工资|工资(?:到账|到手|发放))"
                    r"|(?:工资|薪水|薪资)[^，。！？；]{0,3}(?:还没(?:有)?|尚未|还未|没有)"
                    r"[^，。！？；]{0,4}(?:到|发))",
                    clause,
                )
                if waiting:
                    prefix = clause[: waiting.start()]
                    if not re.search(
                        r"(?:不是|不(?:用|必|需要)?|没(?:有)?|无需)[^，。！？；]{0,4}$", prefix
                    ) and not re.search(r"(?:你|您|他|她|同事|家人|妈妈|爸爸)", prefix + waiting.group(0)):
                        return ["income_note已说明工资到账，不能改说仍需等发工资；保留事实，仅调整表达风格"]
            planned = str(facts.get("planned_date") or "")
            if "月初" in planned and not re.search(r"月底|月末", source_text):
                repayment_action = (
                    r"(?:还(?:款|钱|清|上)|还(?=[零一二三四五六七八九十百千万两\d]|$)"
                    r"|转(?:账|款|过去|进去|入)|支付|付(?:款|钱|清)|交(?:钱|款)|缴(?:费|款))"
                )
                commitment = re.search(
                    r"(?:我(?:这边)?(?:计划|打算|准备|会|就|先)?(?:等到|等|到|在)?"
                    r"(?:这个月|下个月|本月|下月)?(?:月底|月末)[^，。！？；]{0,10}"
                    + repayment_action
                    + r"|(?:月底|月末)我(?:再|就|会|先)?"
                    + repayment_action
                    + r")",
                    clause,
                )
                if commitment and not re.search(r"(?:月底|月末)(?:前|以前|之前)", clause):
                    return ["planned_date给定月初，不能自行承诺月底还款；使用给定日期，不照搬风格举例"]
        return []

    def _unsupported_address_claims(self, text: str) -> list[str]:
        """Check only address assertions when no address is explicitly supplied.

        A standalone phone claim is outside this guard. Questions and uncertainty
        are not address assertions; a later definite filled-history claim still is.
        """
        if not re.search(r"地址|住址", text):
            return []
        sources = {
            "identity": self.user_script.get("identity", ""),
            "user_facts": self.user_script.get("user_facts") or {},
            "db_seed": self.db_seed or {},
        }
        source_text = json.dumps(sources, ensure_ascii=False)
        unknown = r"^(?:未知|未提供|不详|暂无|暂未|待|需|要|还没|不清楚|不知道|稍后|回头|没有|未确认)"
        supplied = re.findall(
            r'"(?:[^"\n]*address[^"\n]*|[^"\n]*(?:地址|住址)[^"\n]*)"\s*:\s*"([^"\n]+)"',
            source_text,
            re.IGNORECASE,
        )
        supplied += re.findall(
            r"(?:地址|住址|家住|住在)(?:是|为|在|[：:])\s*([^，。；;\n\"]{2,})", source_text
        )
        if any(value.strip() and not re.search(unknown, value.strip()) for value in supplied):
            return []
        address_context = False
        for raw_clause in re.findall(r"[^，,。；;！!？?\n]+[，,。；;！!？?]?", text):
            clause = raw_clause.rstrip("，,。；;！!？?").strip()
            if re.search(r"地址|住址", clause):
                address_context = True
            elif re.search(r"电话|号码|联系方式", clause):
                address_context = False
            # Explicit filled-history assertions may refer to an address named in
            # the preceding clause, including a preceding lookup question.
            filled = re.search(
                r"(?:我(?:在)?|下单时|订购时)[^，。！？；]{0,14}(?:填过|填写过|留过|提供过|登记过)"
                r"(?:了|的|的了)?$",
                clause.strip(),
            )
            direct_fill = re.search(r"(?:填过|填写过|留过|提供过|登记过)(?:地址|住址)", clause)
            asking = re.search(r"[？?]", raw_clause) or re.search(
                r"有没有|是否|能否|能不能|可不可以|吗|呢|哪里|哪儿|什么地址", clause
            )
            if asking:
                continue
            if address_context and (filled or direct_fill):
                return ["源资料未给定地址，不能声称地址已填写或已留存；请说明需核实或补充"]
            if not re.search(r"地址|住址", clause):
                continue
            value = re.search(r"(?:地址|住址)(?:就是|是|在|为|[：:])\s*([^，。！？；]{2,})", clause)
            if (
                value
                and not re.search(unknown, value.group(1).strip())
                and not re.search(
                    r"核实|确认|补充|核对|查一下|查到|能查|可以查|帮我查",
                    value.group(1),
                )
            ):
                return ["源资料未给定地址，不能编造具体地址；请说明需核实或补充"]
            stored = re.search(
                r"(?:地址|住址)[^，。！？；]{0,18}(?:系统|记录)[^，。！？；]{0,10}"
                r"(?:已经|已有|已留|本来就|原来就|都有|有记录|有的)",
                clause,
            )
            if stored:
                return ["源资料未给定地址，不能声称系统已有地址；请说明需核实或补充"]
        return []

    def _can_end_call(self, parsed: dict[str, Any]) -> bool:
        text = str(parsed.get("text") or "")
        if _has_termination_cue(text):
            return True
        # Silent/abrupt ending is permitted only by an applicable, explicitly
        # covered script beat after an earlier completed key event. It cannot
        # be inferred from end_call or arbitrary mission completion labels.
        if not self.key_completed or _has_open_request(text):
            return False
        for number in parsed.get("covered") or []:
            if type(number) is not int or not 1 <= number <= len(self.mission):
                continue
            beat = self.mission[number - 1]
            if re.search(r"如果|假如|要是|若|再.{0,10}就", beat):
                continue
            if re.search(r"直接(?:挂断|挂电话|结束通话)|挂断电话|立即挂断", beat):
                return True
        return False

    def _vocab_defense(self, text: str, is_key_turn: bool) -> list[str] | None:
        """校验关键轮文本条件和非关键轮的情绪词约束。

        显式条件关键轮须命中目标状态词；韵律条件与非关键轮禁止明确情绪词。"""
        from listen2serve.vocab import neutral_wording_hits, state_pool

        level = self.leakage_label
        if level == "T3" or (level == "T1" and not is_key_turn):
            hits = neutral_wording_hits(text, is_key_turn=is_key_turn)
            if hits:
                return hits
            return None
        if level == "T1" and is_key_turn:
            pool = state_pool(str(self.key_event.get("state")))
            if pool and not any(w in text for w in pool):
                return [f"未命中 {self.key_event.get('state')} 子表词"]
        return None

    def _assemble_instruct(self, state: str, is_key: bool) -> str:
        """按基础场景和状态读取声音指令；缺失时显式失败。

        neutral 条件始终使用中性声音指令，state 条件使用对应状态指令。"""
        entry = self._instruction_entry  # 构造期已加载并校验，缺失即 raise
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
        """返回声音表指定的标签；中性声音条件不附加状态标签。"""
        if self.prosody_arm == "neutral":
            return ""
        entry = self._instruction_entry
        tag = entry.get("key_tag") if is_key else (entry.get("tags") or {}).get(state)
        return str(tag or "")

    def _tagged_text(self, text: str, state: str, is_key: bool) -> tuple[str, Any]:
        """先附加声音表标签，再校验位置与允许集，保留干净文本和合成文本。"""
        forced = self._forced_tag(state, is_key)
        raw = f"[{forced}]{text}" if forced else text
        allowed = () if self.prosody_arm == "neutral" else self._allowed_tags
        res = apply_tag_policy(raw, allowed=allowed, model=str(self.llm_model or ""), enable_rich=False)
        if res.stripped:
            self.stats["tag_stripped"] += len(res.stripped)
        return forced, res

    def _state_for_turn(self, parsed: dict[str, Any]) -> str:
        """采用模型声明的状态，记录与剧情要点预期状态的偏差。"""
        covered = parsed.get("covered") or []
        expected = self._mission_state_of(covered[-1]) if covered else self._last_state
        state = str(parsed.get("state") or "")
        if expected and state != expected:
            self.stats["state_drift"] += 1
            parsed["_state_expected"] = expected  # 进 llm_debug，供 state 依从率统计
        return state

    def _mission_state_of(self, beat_no: int) -> str:
        """mission 第 beat_no 条（1 基）声明的 state；越界或省略则空串。"""
        if not (1 <= int(beat_no) <= len(self.mission_entries)):
            return ""
        return str(self.mission_entries[int(beat_no) - 1].get("state") or "neutral")

    def _synthesize(self, text: str, state: str, tts_control: str) -> bytes:
        from listen2serve.runtime.tts_normalize import normalize_tts_text

        style = _instruct_from_control(
            state, tts_control, scenario_id=self.scenario_id, turn=self._turn_count + 1
        )
        prosody = _prosody_from_control(tts_control)
        result = self.tts.synthesize(normalize_tts_text(text), voice=self.voice, style=style, prosody=prosody)
        pcm = result.audio
        if self.audio_mode == "regular" or self.effect_config is not None:
            import numpy as np

            arr = np.frombuffer(pcm, dtype=np.int16)
            arr = apply_effects(
                arr,
                result.sample_rate,
                self.effect_config or EffectConfig(),
                np.random.default_rng(self.seed),
            )
            pcm = arr.tobytes()
        return pcm

    def _build_messages(self, n: int, agent_reply: str, *, forced: bool) -> list[dict[str, str]]:
        us = self.user_script
        identity = us.get("identity", "")
        goal = us.get("user_goal") or us.get("goal", "")
        style_block = _render_persona_style(us.get("persona_style"))
        locked = self._initial_disclosure_locks(agent_reply)
        anchor = _render_fact_anchor(self.db_seed)
        for key, value in locked.items():
            for literal in {str(value), _fact_text(key, value)}:
                if re.fullmatch(r"\d+(?:\.\d+)?", literal):
                    anchor = re.sub(
                        r"(?<![A-Za-z0-9_.])" + re.escape(literal) + r"(?![A-Za-z0-9_.])",
                        "（首轮暂不披露）",
                        anchor,
                    )
                else:
                    anchor = anchor.replace(literal, "（首轮暂不披露）")
        facts_block = _render_user_facts(us.get("user_facts"), locked_fields=set(locked))
        disclosure_block = _render_disclosure_plan(self.disclosure_plan)
        if locked:
            disclosure_block += (
                "【首轮披露锁定】尚无客服发言，以下字段本轮暂不提供具体值："
                + "、".join(_USER_FACT_LABELS.get(key, key) for key in locked)
                + "。只说场景允许的一般诉求，等待客服按原披露条件询问；"
                "之后有客服发言也不代表复合条件自动满足，仍按原披露时机推进。\n"
            )
        ke = self.key_event
        level = self.leakage_label or "T3"
        neutral_key_view = level == "T3" and bool(ke.get("surface"))
        displayed_mission = list(self.mission)
        if neutral_key_view:
            for i, entry in enumerate(self.mission_entries):
                if entry.get("key_event"):
                    displayed_mission[i] = f"【关键事件】{ke['surface']}"
        mission_block = _render_mission(displayed_mission)
        goal_block = f"【目标】{goal}\n"
        if neutral_key_view:
            goal_block = (
                f"【内部剧情目标】{goal}（仅供内部剧情判断，不作为用户台词；"
                "text仍按当前业务事实与中性关键轮表面话题表达。）\n"
            )
        covered = sorted(self._covered)
        progress = "、".join(str(c) for c in covered) if covered else "无"
        key_status = f"已在第 {self._key_turn_no} 轮完成" if self.key_completed else "未完成"
        key_block = (
            f"【关键事件】{ke.get('description', '')}（要求状态：{ke.get('state')}，"
            f"触发时机：{ke.get('trigger_hint', '自然推进时')}；"
            f"当前状态：{key_status}）\n"
        )
        if neutral_key_view:
            key_block = (
                f"【关键事件】表面业务话题：{ke['surface']}；业务目标：{ke.get('objective', '')}。"
                f"触发时机：{ke.get('trigger_hint', '自然推进时')}；当前状态：{key_status}。\n"
                "【中性关键轮文本边界】关键轮text以表面业务话题为唯一措辞依据，保持中性的事实核对或业务询问；"
                "业务目标用于维持剧情方向，不据此追加时间压力、态度结论、状态自述或施压要求。"
                "这一措辞边界优先于剧情要点、关键事件描述、身份和风格示范中的态度表达；"
                "真实意图和情绪只用于内部剧情、JSON的state及声音演绎，不要把它们说进text。"
                f"发生关键事件的那一轮JSON state仍为{ke.get('state')}，该关键轮声音继续按此状态演绎。\n"
            )
        transparency = _transparency_clause(level, self._t1_sample_words)
        budget_line = f"【轮次预算】整通电话争取在 {self.soft_budget} 轮内办完你要办的事；当前第 {n} 轮。\n"
        if not self.key_completed:
            if forced:
                budget_line += "【预算强制】本轮必须完成【关键事件】（is_key_turn=true）。\n"
            elif n >= self.soft_budget:
                budget_line += "【预算告警】关键事件尚未完成，请尽快收尾，最迟下一轮必须完成【关键事件】。\n"
        elif n >= self.soft_budget:
            budget_line += (
                "【预算告警】关键事件已完成不等于事情已办完；继续回应当前事项并推进仍适用的剧情，"
                "真正收束后再道别（end_call=true），在硬预算内结束。\n"
            )
        forbidden = us.get("forbidden_user_behavior", [])
        forbidden_block = _render_forbidden(forbidden)
        examples = _fewshot_examples(self.role, self.dynamics)
        fewshot_block = ""
        if examples:
            fewshot_block = (
                "【风格示范】以下口语短句仅作语气/节奏参考，严禁照抄内容：\n"
                + "\n".join(f"- {e}" for e in examples)
                + "\n"
            )
        history_block = _render_history(self._history)
        system = (
            "你正在与客服进行一通真实的中文电话通话，扮演打/接电话的用户。\n"
            "你只扮演【用户】一方：绝不能说客服的台词，绝不能重复或改写客服刚才说的话。\n"
            "每轮必须先承接客服上一轮的问题或话题（先回应、再推进），不得跳过客服的话题自说自话；\n"
            "收尾轮先用一短句回应客服刚提出的事项，再道别，不得说完自己的事立刻挂断；"
            "客服登记姓名、联系方式、地址或可用时间时，只能提供场景已明确给定的信息；"
            "字段未提供或无法确认，就说明需要核实或补充，或询问安全补充方式，不得为配合登记编造具体值，"
            "也不得凭空声称系统已有这些信息。回应已知事项后再决定是否收束。\n"
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
            "还必须有实际道别、改约或拒绝继续通话的收束语，或已执行脚本明确允许的直接挂断；"
            "普通业务回复、仅确认已知信息或仍在问问题时 end_call=false，关键轮完成本身不代表通话结束。"
        )
        stable_prefix = (
            f"【身份】{identity}\n"
            f"{style_block}"
            f"{goal_block}"
            f"【事实锚】{anchor}\n"
            f"{facts_block}"
            f"{disclosure_block}"
            "【事实优先】身份、系统事实锚和用户私有事实决定业务内容；说话风格只影响表达，"
            "不得照搬风格中的发薪、金额或日期举例来替换事实；事实冲突时省略该风格举例。\n"
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
        closing_feedback = (
            "【结束请求反馈】上轮end_call未获准，通话仍在继续；先承接客服当前回应。"
            "普通业务回复或条件性投诉不代表已经收束；需要结束时按场景真实道别、改约、拒绝继续，"
            "或执行已适用的直接挂断，不要机械重复上一句。\n"
            if self._last_end_call_rejected
            else ""
        )
        dynamic_tail = (
            f"【进度】已完成要点：{progress}\n"
            f"{closing_feedback}"
            f"{budget_line}"
            f"【完整对话历史】\n{history_block}\n"
            f"【客服上轮回复】{agent_reply or '（对话开始，无上轮）'}\n"
            "【要求】自然承接客服上文，按协议输出紧凑 JSON；covered 里报出本轮完成的要点序号。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": stable_prefix + dynamic_tail},
        ]


_EXPLICIT_TERMINATION = (
    *_USER_CLOSING,
    "改天再联系",
    "改日再联系",
    "下次再联系",
    "以后再联系",
    "回头再联系",
    "改天再联系我",
    "改日再联系我",
    "下次再联系我",
    "以后再联系我",
    "回头再联系我",
    "改天再打",
    "改日再打",
    "下次再打",
    "以后再打",
    "别再打过来",
    "不要再打过来",
    "别再打了",
    "不要再打了",
    "今天先到这",
    "先说到这",
    "先到这里",
    "就这么定了",
)

_WAITING_ACKNOWLEDGMENT = re.compile(
    r"(?:我|我们)(?:就|先|会)?(?:等|等着|等待)"
    r"(?:(?:你们|您|你|专员|客服|售后|平台)(?:的)?)?"
    r"(?:电话|回电|回访|短信(?:通知)?|消息|通知)(?:就行|就好|即可|吧|了)?$"
)


def _has_open_request(text: str) -> bool:
    if re.search(r"[？?]|(?:吗|呢|么)[。！!]*$", text):
        return True
    if re.search(r"(?:怎么|如何|多少|哪(?:天|里|个|种))[^，。！？；]{0,12}$", text) and not re.search(
        r"(?:已经|早已|都)[^，。！？；]{0,3}(?:知道|明白|清楚|确认|了解)(?:了)?[。！!]*$",
        text,
    ):
        return True
    if re.search(
        r"(?:别|不要|先别)[^，。！？；]{0,3}(?:挂|结束通话)|(?:还需要|还要|还得|还没|尚未)"
        r"[^，。！？；]{0,8}(?:确认|核对|问清|说清)",
        text,
    ):
        return True
    return bool(
        re.search(
            r"(?:请|麻烦|劳驾|你|您)[^，。！？；]{0,25}(?:说|告诉|解释|确认|核对|介绍|说明|发|提供|给我)"
            r"[^，。！？；]*(?:一下|吧|清楚|方式|时间|期限|渠道)[。！!]*$",
            text,
        )
    )


def _has_termination_cue(text: str) -> bool:
    if re.search(r"[？?]|(?:吗|呢|么)[。！!]*$", text):
        return False
    if _has_open_request(text) and not re.search(r"(?:再见|拜拜)[吧啦了啊呀哈哦]*[。！!]*$", text):
        return False
    for clause in re.split(r"[，,。；;！!\n]", text):
        if re.search(r"如果|假如|要是|再.{0,10}就", clause):
            continue
        if _WAITING_ACKNOWLEDGMENT.search(clause.strip()) and not re.search(
            r"(?:不是|不(?:要|能|想|打算|用|必|需要)|还没|没说|别)[^，。！？；]{0,3}"
            r"(?:我|我们)(?:就|先|会)?(?:等|等着|等待)",
            clause,
        ):
            return True
        for cue in _EXPLICIT_TERMINATION:
            match = re.search(re.escape(cue) + r"[吧啦了啊呀哈哦]*$", clause.strip())
            if match and not re.search(
                r"(?:不是|不(?:要|能|想|打算|用|必|需要)|还没|没说|别)[^，。！？；]{0,3}$",
                clause[: match.start()],
            ):
                return True
    return False


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
        data = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _render_persona_style(persona_style: list[Any] | None) -> str:
    """persona 风格卡（口头禅频率封顶条款保留）。"""
    if not persona_style:
        return ""
    items = [str(s) for s in persona_style if str(s).strip()]
    if not items:
        return ""
    return (
        f"【说话风格】你的说话风格：{'；'.join(items)}。风格服从于已给定事实、透明度约束与剧情需要，"
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
    "debt_amount": "欠款金额",
    "overdue_days": "逾期天数",
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


def amount_to_spoken_zh(value: float) -> str:
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
        head = head.replace("一十", "十", 1) if head.startswith("一十") else head
    if frac:
        digits = f"{frac:.2f}".split(".")[1].rstrip("0")
        head += "点" + "".join(_ZH_DIGITS[int(d)] for d in digits)
    return ("负" if negative else "") + head


def _fact_text(key: str, value: Any) -> str:
    if key == "planned_amount" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{amount_to_spoken_zh(value)}元"
    return str(value)


def _finance_number_tokens(text: str) -> list[str]:
    """Compare whole amount tokens, including Arabic/Chinese presentation."""
    text = re.sub(
        r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9])",
        lambda match: amount_to_spoken_zh(float(match.group().replace(",", ""))),
        text,
    )
    return re.findall(r"[零〇一二两三四五六七八九十百千万亿点]+", text)


def _finance_amount_check(text: str) -> bool:
    """Recognize a debt quotation/question, including an incorrect quotation.

    Target correctness remains a judge concern; the user may correct the amount.
    """
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    # Explicit non-money fields do not become debt quotations merely because
    # they contain a number and the assistant asks for confirmation.
    digits = r"[\d零〇一二两三四五六七八九十百千万亿点]+"
    text = re.sub(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", "", text)
    text = re.sub(digits + r"\s*(?:天|年|月|日|号|点|时|期|次)", "", text)
    text = re.sub(r"(?:联系电话|电话|手机号|工号|订单号|编号)\s*" + digits, "", text)
    quoted_number = re.search(r"\d|[零〇一二两三四五六七八九十百千万亿点]+(?:元|块)|[十百千万亿]", text)
    total_object = re.search(r"欠款|欠费|应还总额|债务|债额|总额", text)
    if re.search(r"首期|每期|本期|先还|一部分", text) and not total_object:
        return False
    money_unit = re.search(r"[\d零〇一二两三四五六七八九十百千万亿点]+\s*(?:元|块)", text)
    explicit_money_question = money_unit and not re.search(r"还|缴|交|转|付", text)
    if (
        quoted_number
        and explicit_money_question
        and re.search(r"(?:您|你|请|麻烦).{0,6}(?:确认|核对)|[？?]", text)
    ):
        return True
    for clause in re.split(r"[，,。；;！!\n]", text):
        if not re.search(r"欠款|欠费|应还|金额|数额|总额|这笔", clause):
            continue
        if re.search(
            r"(?:多少|几元|几块|确认|核对).{0,8}(?:金额|数额|总额)|(?:金额|数额|总额).{0,8}(?:多少|几元|几块|对吗|是吗)",
            clause,
        ) and re.search(r"多少|几元|几块|对吗|是吗", clause):
            return True
        if re.search(r"(?:尚未|还没|未|没有|不确定).{0,6}(?:核对|确认)|不确定|(?:假设|举例)", clause):
            continue
        if re.search(r"首期|每期|本期|先还|一部分", clause) and not re.search(
            r"(?:欠款(?:金额|总额)?|欠费|应还总额|总欠)(?:是|为|有|一共|共计)?\s*" + digits, clause
        ):
            continue
        if _finance_number_tokens(clause):
            return True
    return False


def _finance_prerequisite_evidence(spoken: list[str]) -> dict[str, list[str]]:
    """Extract bounded business-action evidence, not emotion or answer quality."""
    found: dict[str, list[str]] = {"影响说明": [], "官方渠道": [], "还款协商": []}
    for text in spoken:
        for raw in re.findall(r"[^，,。；;！!？?\n]+[，,。；;！!？?]?", text):
            clause = raw.strip()
            question = bool(re.search(r"[？?]|是否|是不是|不确定|不知道", clause))
            informs_by_question = bool(
                re.search(r"(?:您|你)(?:是否)?(?:知道|清楚|了解).*(?:会|将).*(?:影响|产生|增加|上报)", clause)
            ) and not re.search(r"是否会|是不是会|会不会|不确定", clause)
            if (
                (not question or informs_by_question)
                and not re.search(r"(?:尚未|还没|未|没有).{0,6}(?:说明|解释|告知)", clause)
                and (
                    re.search(r"征信|信用记录|罚息|滞纳金|服务", clause)
                    and re.search(r"影响|上报|报送|产生|增加|累计|收取|限制|暂停|停用", clause)
                )
            ):
                found["影响说明"].append(clause)
            channel = re.search(
                r"官方\s*(?:APP|应用|渠道|入口)|平台指定(?:的)?还款入口", clause, re.IGNORECASE
            )
            payment_use = re.search(r"还款(?!信息|明细|记录)|还钱|还清|缴费|缴款|转账", clause)
            private_account_contrast = re.search(r"私转|私人(?:转账|账户|账号)", text)
            if (
                channel
                and (payment_use or private_account_contrast)
                and not question
                and not re.search(
                    r"(?:不是|并非|不属于|非|不用|不要用)(?:我们的|本平台的|这个|该)?\s*$",
                    clause[: channel.start()],
                )
                and not re.search(r"(?:尚未|还没|未|不能).{0,4}(?:确认|核实)", clause)
                and not re.match(
                    r"\s*(?:并)?(?:不是|并非|不属于|非)(?:我们的|本平台的|这个|该)?(?:还款)?(?:渠道|入口|应用|APP)",
                    clause[channel.end() :],
                    re.IGNORECASE,
                )
            ):
                found["官方渠道"].append(clause)
            if re.search(r"(?:后续|以后|之后|稍后|改天|回头).{0,5}(?:再)?(?:协商|商量|讨论|安排)", clause):
                continue
            if re.search(r"(?:尚未|还没|未|没有).{0,4}(?:协商|商量|安排)", clause):
                continue
            repay = (
                r"还钱|还清|还款(?!信息|明细|记录|渠道|入口)|缴费|缴款|交钱|交款"
                r"|还(?:个)?(?=多少|一部分|[零一二两三四五六七八九十百千万\d吗呢吧？?。]|$)"
            )
            if re.search(r"(?:不用|无需|不需要|不要|暂不).{0,4}(?:" + repay + r")", clause):
                continue
            if (
                re.search(repay, clause)
                and re.search(
                    r"什么时候|何时|哪天|几号|多少|能不能|能否|方便|打算|安排|计划|考虑|您看|先还|一次性|分期|一部分|今天|明天|月初|月底",
                    clause,
                )
            ) or re.search(r"(?:可以|可选|考虑|一次性|还是).{0,6}分期", clause):
                found["还款协商"].append(clause)
    return found


def _finance_executable_plan(text: str) -> bool:
    """Recognize an offered payment option within a clause, not a generic request.

    A concrete amount, partial/full payment option or installment application is
    enough. Correct amounts, affordability and approval authority remain judge
    concerns. Debt quotations elsewhere cannot supply a proposal's amount.
    """
    number = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)"
    pay = r"(?:还款|偿还|归还|还(?:个)?|缴纳|缴费|缴款|交(?:个)?|支付|付)"
    for clause in re.split(r"[，,。；;！!？?\n]|(?:但是|但|不过|(?:那)?就)(?=先|可以|可|请)", text):
        if re.search(
            r"(?:上次|上回|此前|之前|曾经|昨天|昨日|已经|已).{0,4}" + pay + r"(?:了)?\s*" + number, clause
        ):
            continue
        if re.search(
            r"(?:后续|以后|之后|稍后|改天|回头).{0,5}(?:再)?(?:讨论|协商|商量|安排|提供|申请)", clause
        ):
            continue
        if re.search(
            r"(?:不能|无法|暂不|不支持|不可以|不提供|尚未|还没|没有).{0,6}(?:申请|提供|提交|分期|部分|全额)",
            clause,
        ):
            continue
        if re.search(r"(?:不用|无需|不需要|不要|暂不).{0,4}" + pay, clause):
            continue
        if re.search(
            r"(?:部分|分期|全额|方案|安排|方式|做法|选项|这样).{0,5}"
            r"(?:不允许|不可以|不可|不行|行不通|不接受|不支持|无法执行)"
            r"|" + pay + r"\s*" + number + r"(?:元|块)?\s*(?:也)?(?:不行|不可以|不允许)",
            clause,
        ):
            continue
        if re.search(
            r"(?:可以|可|能|支持|帮您|为您|先|请).{0,8}(?:申请|提交|办理).{0,5}分期"
            r"|分期.{0,5}(?:可以|可|能).{0,3}申请"
            r"|(?:可以|可选|考虑|直接|支持).{0,5}分期"
            r"|分" + number + r"期",
            clause,
        ):
            return True
        if re.search(pay + r"\s*" + number, clause):
            return True
        if re.search(
            pay + r"(?:一)?部分|先(?:还|交|缴)(?:点|一点|一部分|部分)|部分(?:还款|缴纳|缴费)", clause
        ):
            return True
        if (
            re.search(r"全额|全部|一次性|还清|结清", clause)
            and re.search(pay, clause)
            and re.search(r"请|可以|建议|一次性|今天|明天|通过|或者|也可以", clause)
        ):
            return True
    return False


def _render_user_facts(user_facts: dict[str, Any] | None, *, locked_fields: set[str] | None = None) -> str:
    if not user_facts:
        return ""
    lines = []
    for key, value in user_facts.items():
        label = _USER_FACT_LABELS.get(key, key)
        shown = (
            "（首轮暂不披露；等待客服按披露时机询问）"
            if key in (locked_fields or ())
            else _fact_text(key, value)
        )
        lines.append(f"- {label}：{shown}")
    return "【用户本人掌握的事实】\n" + "\n".join(lines)


def _render_disclosure_plan(plan: dict[str, Any] | None) -> str:
    """注明用户私有事实的披露时机。"""
    if not plan:
        return ""
    lines = []
    for key, when in plan.items():
        label = _USER_FACT_LABELS.get(key, key)
        lines.append(f"- {label}：{when}")
    return (
        "【披露时机】以下信息是你的底牌，时机未到绝不主动透露，被问到且时机已到才说；"
        "透露时先用半句话承接客服上一轮（如‘你问我还款能力，实际情况是…’），不得凭空抛出数字：\n"
        + "\n".join(lines)
        + "\n"
    )


_AFTER_HINT = {
    "agent_states_purpose": "等客服说明来意之后再做这一条，别自己先把话题点出来",
}


def _render_mission(mission: list[str] | list[dict[str, Any]]) -> str:
    """将剧情要点及其 after 前置条件渲染为列表。"""
    lines = []
    for i, m in enumerate(mission, 1):
        beat = m if isinstance(m, str) else str(m.get("beat") or "")
        after = "" if isinstance(m, str) else str(m.get("after") or "")
        hint = _AFTER_HINT.get(after, "")
        lines.append(f"{i}. {beat}（{hint}）" if hint else f"{i}. {beat}")
    return "【剧情要点】（整通电话你要办成的事，顺序可自行调整，无需逐条念出）\n" + "\n".join(lines) + "\n"


def _transparency_key_clause(level: str, t1_words: list[str]) -> str:
    """根据关键轮文本条件渲染情绪词要求。"""
    if level == "T1":
        if t1_words:
            sample = "/".join(t1_words)
            return f"须自然带出至少 1 个该状态的明确情绪词（参考词：{sample}）。"
        return "按关键事件要求状态自然表达。"
    if level == "T3":
        return "仍用纯业务中性词汇表达，禁止任何情绪/态度词。"
    raise ValueError(f"未知文本条件：{level!r}")


def key_form_for(scenario_id: str) -> str:
    """按基础场景编号奇偶返回 question 或 statement；同一场景各条件一致。"""
    base = re.sub(r"-T[123]$", "", scenario_id or "")
    m = re.search(r"(\d+)$", base)
    if not m:
        return "question"
    return "statement" if int(m.group(1)) % 2 == 0 else "question"


# 供文本线索诊断使用；中性文本中的立场结论标记。
NEUTRAL_STANCE_MARKERS = (
    "不需要",
    "不用了",
    "不要再",
    "不打算",
    "不接受",
    "没法接受",
    "等不起",
    "太久",
    "别打",
    "别再",
    "处理不了",
    "拿不出",
    "没法配合",
    "不用你们管",
    "行吧",
    "算了",
    "好吧",
    "那行",
    "无所谓",
    "随便吧",
)


def _transparency_clause(level: str, t1_words: list[str]) -> str:
    """非关键轮使用克制中性文本，关键轮按显式或中性文本条件约束。"""
    return "非关键轮：克制中性，禁止明确情绪词。轮到【关键事件】时：" + _transparency_key_clause(
        level, t1_words
    )


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


def _fewshot_examples(role: str, dynamics: str) -> list[str]:
    return _FEWSHOT_EXAMPLES.get((role, dynamics)) or _FEWSHOT_EXAMPLES.get(("", dynamics), [])


def _render_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "（无）"
    lines = []
    for m in history:
        who = "用户" if m["role"] == "user" else "客服"
        lines.append(f"{who}: {m['content']}")
    return "\n".join(lines)


_LEVEL_BY_NAME = {
    "T1": "T1",
    "explicit": "T1",
    "T3": "T3",
    "prosody_only": "T3",
}


def normalize_leakage_level(value: str) -> str:
    """将公开文本条件归一为资产标签 T1/T3；无法识别返回空串。"""
    return _LEVEL_BY_NAME.get(str(value or "").strip(), "")


_LEAKAGE_SUFFIX_RE = re.compile(r"-T[13]$")


def _base_scenario_id_of(scenario_id: str) -> str:
    """去掉文本条件尾缀，得到双方条件共用的声音指令索引。"""
    return _LEAKAGE_SUFFIX_RE.sub("", str(scenario_id or "").strip())


_ROLE_IDENTITY_WORD = {
    "collection": "欠款客户",
    "hotline": "来电客户",
    "marketing": "意向客户",
}


def _instruct_from_control(state: str, tts_control: str, *, scenario_id: str = "", turn: int = 0) -> str:
    """校验声音指令非空且包含语速或音量描述，再添加合成包装。"""
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
    """提取语速和音量的辅助记录。实际韵律由 TTS instruct 文本控制。"""
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
