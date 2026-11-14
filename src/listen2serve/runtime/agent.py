"""被测客服封装：按 tick 处理音频原生全双工对话。

系统提示词组合共享通话指令与场景策略。工具仅在显式启用并注入域数据库时执行；
未启用的调用仍保留记录，并返回禁用提示。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from listen2serve.domains.base import (
    AGENT_CLOSING_RULE,
    AGENT_DUAL_RESPONSE_RULE,
    AGENT_STATE_FIRST_RULE,
    AGENT_VOICE_STATE_RULE,
    STATE_LABELS,
    DomainSpec,
    action_table,
)
from listen2serve.runtime.realtime.registry import create_provider
from listen2serve.runtime.realtime.tick_adapter import QwenTickAdapter

logger = logging.getLogger(__name__)

_STATE_MENU = "／".join(
    STATE_LABELS[k] for k in ("cooperative", "displeased", "urgent", "doubtful", "hushed")
)

_VOICE_CALL_BODY = f"""
# 本次通话的第一要务（优先级高于下方任何业务规范）

你正在与客户进行一通**真实的电话通话**，是在"说话"而不是"打字"。
这通电话考察的核心能力是：**从客户的声音里判断他此刻的状态，并据此选择正确的服务动作。**

## 一、先听状态，再决定动作
1. {AGENT_VOICE_STATE_RULE}
2. 客户每说完一句，你先判断他此刻处于哪种状态（{_STATE_MENU}）。客户没有把情绪说出口时，
   不要默认他是中立或配合。

## 二、听出状态后必须双通道应答（本次通话的评分重心）
3. {AGENT_DUAL_RESPONSE_RULE}
4. 不要把判断当结论报出来（不说"我检测到您情绪不佳""听出您有点着急"这类话），
   要把判断落到你实际做的事情和你的声音上：该停就停、该压缩就压缩、该自证就自证、该放软就放软。
   自然的共情表达（"您别急""理解您的顾虑"）是可以的。

## 三、状态应对优先于流程推进
5. {AGENT_STATE_FIRST_RULE}

## 四、必须有收尾轮
6. {AGENT_CLOSING_RULE}

## 五、说话方式
7. 像真人客服一样自然、口语化地说话：允许语气词（嗯/好的/您说）、短暂停顿、
   自然的语速起伏；不要像朗读稿件，不要机械地逐条念出。
8. 回复不要使用列表、编号、标题或任何书面格式，用连贯的口语句子表达。
9. 客户的声音可能受电话信道影响听不清，必要时请客户重复或确认。
10. 保持自然对话节奏：客户说完后稍作承接再进入正题，不要每次都用同一句式开头。
""".strip()

_STATE_ACTION_TABLE = action_table(
    heading="## 六、状态→动作速查表（唯一动作口径：听出哪一种，下一轮就先做哪一行的开场，再接一句跟进）"
)

VOICE_CALL_INSTRUCTION = _VOICE_CALL_BODY + "\n\n" + _STATE_ACTION_TABLE


def voice_call_instruction() -> str:
    """返回正式多轮评测共用的通话指令。"""
    return VOICE_CALL_INSTRUCTION


TOOL_DISABLED_OUTPUT = '{"error": "tool execution disabled in this run"}'


@dataclass
class AgentTickResult:
    """Agent 一个 tick 的输出（编排器消费）。"""

    audio: bytes = b""  # 本 tick 播出的客服音频（24k PCM16，未填充）
    transcript: str = ""  # 与本 tick 播出音频成比例的转写增量
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # 含 name/arguments/call_id
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    response_done: bool = False
    done_status: str = ""  # response.done 携带的状态（completed/cancelled/failed/incomplete）
    speech_started: bool = False  # server VAD 打断信号
    was_truncated: bool = False
    truncated_bytes: int = 0
    has_activity: bool = False
    agent_output: bool = False  # 本 tick 收到下行产出事件（音频/转写/done）——供编排器判轮
    buffer_bytes: int = 0  # 适配器中尚未播出的缓冲音频
    tools_flushed: int = 0  # 本 tick 投递的工具结果数
    errors: list[str] = field(default_factory=list)
    # None 表示旧适配器没有 item 身份能力；空列表表示有能力但该 tick 无已知 item。
    played_item_ids: list[str] | None = None
    truncated_item_ids: list[str] | None = None


class Agent:
    """被测 Agent（音频原生 Realtime，tick 级全双工）。"""

    def __init__(
        self,
        model_spec: str,
        api_key: str,
        domain: DomainSpec,
        system_prompt: str,
        voice: str | None = None,  # None = 取模型绑定表音色（registry 内解析）
        tick_ms: int = 200,
        vad_mode: str = "server_vad",
        tools_enabled: bool = False,  # False（默认）：start() 不注册工具（评测不发起 function call）
    ) -> None:
        self.model_spec = model_spec
        self._api_key = api_key
        self._voice = voice  # 故障转移时要按同一音色重建 provider，故必须留住
        self.tick_ms = tick_ms
        self.provider = create_provider(model_spec, api_key, voice=voice)
        self.adapter = QwenTickAdapter(self.provider, tick_ms=tick_ms)
        self.domain = domain
        self.system_prompt = system_prompt
        self.vad_mode = vad_mode
        self.tools_enabled = tools_enabled
        self.domain_db = None  # 工具执行预留：attach_db 注入后才启用

    async def start(self) -> None:
        tools = (
            []
            if not self.tools_enabled
            else [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in self.domain.tools
            ]
        )
        full_prompt = f"{voice_call_instruction()}\n\n{self.system_prompt}"
        try:
            await self.adapter.connect(
                system_prompt=full_prompt,
                tools=tools,
                vad_mode=self.vad_mode,
            )
        except Exception as exc:  # 只为换 key 后重试一次；换不动就原样抛，不吞错
            if not self._switch_volc_key(exc):
                raise
            await self.adapter.connect(
                system_prompt=full_prompt,
                tools=tools,
                vad_mode=self.vad_mode,
            )

    def _switch_volc_key(self, exc: BaseException) -> bool:
        """火山后端连接失败时切换到配置中的下一把 Key，并重建同参数适配器。

        不按供应商错误文案猜测故障类别；没有可用替代 Key 时保留原始错误。
        """
        from listen2serve.model_registry import resolve_model
        from listen2serve.runtime.config import get_settings

        try:
            backend, _name = resolve_model(self.model_spec, "")
        except Exception:  # noqa: BLE001 - 解析不出来就不是 volc，走原错误
            return False
        if backend != "volc":
            return False
        s = get_settings()
        nxt = s.volc_mark_exhausted(self._api_key)
        if not nxt:
            return False
        logger.warning(
            "豆包 realtime 连接失败，切换到下一把已配置的 API Key（错误类型：%s）",
            type(exc).__name__,
        )
        self._api_key = nxt
        self.provider = create_provider(self.model_spec, nxt, voice=self._voice)
        self.adapter = QwenTickAdapter(self.provider, tick_ms=self.tick_ms)
        return True

    async def stop(self) -> None:
        await self.adapter.disconnect()

    def attach_db(self, db) -> None:
        """注入域 DB，启用工具执行（预留机制：未注入时工具返回禁用提示）。"""
        self.domain_db = db

    @property
    def provider_identity(self) -> dict[str, Any]:
        """Requested friendly name, actual wire request and provider-reported model."""
        identity = dict(getattr(self.provider, "identity", {}) or {})
        identity["requested_wire_model"] = identity.pop("requested_model", None)
        identity["requested_model"] = self.model_spec
        identity.setdefault("response_model", None)
        return identity

    @property
    def session_usage(self) -> dict[str, int]:
        """本会话累计 Realtime token 用量（审计/熔断用）。"""
        return self.adapter.session_usage

    @property
    def needs_continuous_audio(self) -> bool:
        """端点是否要求连续上行音频；编排器据此保持静音流并调整等待预算。"""
        return bool(getattr(self.provider, "needs_continuous_audio", False))

    def stream_diag(self) -> dict[str, Any]:
        """端点自定义的上行流诊断（豆包：静音保活补发帧数）；无此能力的端点返回空。"""
        fn = getattr(self.provider, "stream_diag", None)
        return dict(fn()) if callable(fn) else {}

    async def run_tick(self, user_pcm16: bytes, tick: int = 0) -> AgentTickResult:
        """运行一个 tick；工具调用当场执行并将结果排队至下一 tick 投递。"""
        r = await self.adapter.run_tick(user_pcm16, tick)
        played_ids = getattr(r, "played_item_ids", None)
        truncated_ids = getattr(r, "truncated_item_ids", None)
        result = AgentTickResult(
            audio=r.agent_audio,
            transcript=r.transcript,
            response_done=r.response_done,
            done_status=r.done_status,
            speech_started=r.speech_started,
            was_truncated=r.was_truncated,
            truncated_bytes=r.truncated_bytes,
            has_activity=r.has_activity,
            agent_output=r.agent_output,
            buffer_bytes=r.buffer_bytes,
            tools_flushed=r.tools_flushed,
            errors=list(r.errors),
            played_item_ids=list(played_ids) if played_ids is not None else None,
            truncated_item_ids=list(truncated_ids) if truncated_ids is not None else None,
        )
        for fc in r.tool_calls:
            args = _parse_args(fc.get("arguments", "{}"))
            record = {"call_id": fc["call_id"], "name": fc["name"], "arguments": args}
            result.tool_calls.append(record)  # 记录始终保留（评测/审计用）
            if self.domain_db is not None:
                output = self.domain.execute_tool(self.domain_db, fc["name"], args)
            else:
                output = TOOL_DISABLED_OUTPUT
            result.tool_outputs.append({**record, "output": output})
            self.adapter.queue_tool_result(fc["call_id"], output)
        return result

    async def cancel_response(self) -> None:
        await self.adapter.cancel_response()


def _parse_args(arguments: str) -> dict[str, Any]:
    import json

    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
