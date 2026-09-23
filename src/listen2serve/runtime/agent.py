"""被测 Agent 封装：离散时间音频原生对话（tick 级全双工）。

职责：
- 持有 Realtime Provider（Qwen WSS）与 tick 适配器
- 装配系统提示（策略指令）+ 工具集（tools_enabled 开关控制：2026-08-14 起默认
  不注册工具，评测不发起 function call，需查询的用户信息改注入初始 prompt）
- 每 tick 接收用户音频块 → 返回本 tick 客服音频/转写/工具调用
- 工具执行：**预留机制**——仅当注入了域 DB（attach_db）时真正执行；
  未启用时返回禁用提示，但工具调用记录完整保留（name/arguments/call_id）
"""

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
    resolve_agent_prompt_version,
)
from listen2serve.runtime.realtime.registry import create_provider
from listen2serve.runtime.realtime.tick_adapter import QwenTickAdapter

logger = logging.getLogger(__name__)

# 电话通话自然度指令（借鉴 tau-voice 的 AUDIO_NATIVE_VOICE_INSTRUCTION 设计：
# 明确"VOICE CALL"上下文、口语化、禁用书面格式、容忍转写噪音）
VOICE_CALL_INSTRUCTION_V6 = """
# 电话通话注意事项（重要）

1. 你正在与客户进行一通**真实的电话通话**，是在"说话"而不是"打字"。
2. 像真人客服一样自然、口语化地说话：允许语气词（嗯/好的/您说）、短暂停顿、
   自然的语速起伏；不要像朗读稿件，不要机械地逐条念出。
3. 回复不要使用列表、编号、标题或任何书面格式，用连贯的口语句子表达。
4. 客户的声音可能受电话信道影响听不清，必要时请客户重复或确认。
5. 保持自然对话节奏：客户说完后稍作承接再进入正题，不要每次都用同一句式开头。
""".strip()

# v7 状态清单：与 STATE_LABELS 同源（勿在 prompt 里另写一份标签）
# 2026-09-02 用户裁决：Step2 探针与 agent prompt 中状态判断改为 5 选 1，去掉 neutral，
# 避免 neutral 成为"不确定时垃圾桶"，同时压掉文字/对照臂的干扰。
_STATE_MENU = "／".join(
    STATE_LABELS[k] for k in ("cooperative", "displeased", "urgent", "doubtful", "hushed")
)

# v7（本基准正式任务口径）：把「听出状态 → 做对动作」摆在第一位，并补齐 v6 缺失的
# 两条会话级规则（状态应对优先于流程推进、强制收尾轮）。三条核心规则文本取自
# domains.base 常量，与裁判契约逐字节同源。
VOICE_CALL_INSTRUCTION_V7 = f"""
# 本次通话的第一要务（优先级高于下方任何业务规范）

你正在与客户进行一通**真实的电话通话**，是在"说话"而不是"打字"。
这通电话考察的核心能力是：**从客户的声音里判断他此刻的状态，并据此选择正确的服务动作。**

## 一、先听状态，再决定动作
1. {AGENT_VOICE_STATE_RULE}
2. 客户每说完一句，你先判断他此刻处于哪种状态（{_STATE_MENU}）。客户没有把情绪说出口时，
   不要默认他是中立或配合。
3. 判断出状态后，按下方「用户状态应对策略」执行该状态对应的必须动作、避开禁止动作。

## 二、听出状态后必须双通道应答（本次通话的评分重心）
4. {AGENT_DUAL_RESPONSE_RULE}
5. 不要把判断当结论报出来（不说"我检测到您情绪不佳""听出您有点着急"这类话），
   要把判断落到你实际做的事情和你的声音上：该停就停、该压缩就压缩、该自证就自证、该放软就放软。
   自然的共情表达（"您别急""理解您的顾虑"）是可以的。

## 三、状态应对优先于流程推进
6. {AGENT_STATE_FIRST_RULE}

## 四、必须有收尾轮
7. {AGENT_CLOSING_RULE}

## 五、说话方式
8. 像真人客服一样自然、口语化地说话：允许语气词（嗯/好的/您说）、短暂停顿、
   自然的语速起伏；不要像朗读稿件，不要机械地逐条念出。
9. 回复不要使用列表、编号、标题或任何书面格式，用连贯的口语句子表达。
10. 客户的声音可能受电话信道影响听不清，必要时请客户重复或确认。
11. 保持自然对话节奏：客户说完后稍作承接再进入正题，不要每次都用同一句式开头。
""".strip()

# v8（⚠️ 已废弃：**含 variant 透传缺陷，仅供历史 run 复算，禁止用于新实验** —— 新实验用 v9）。
# 当年的意图是「在 v7 之上追加『状态→动作速查表』，其余同 v7」（用户裁决 2026-09-02：给客服
# 明确、合理的 state-动作指示）。**实际不是「其余同 v7」**：domains/base.py 的
# render_policy_sections 当年把精简分支判断写成 `== "v7"`，v8 落不进去、掉进 v6 全量分支，
# 于是 # 通话目标与业务规范 板块把 v7 有意剔除的东西全灌了回来（业务流程六阶段展开、
# 「## 说话风格」整节、流程顺序红线、「用户类型应对要点」知识条）。
# 数量对账（样例 MAR-ALL-03-T3）：v8 − v7 = 5016 − 3410 = 1606 字，下面这张速查表只占
# 281 字（1505 − 1224），剩下 1325 字是缺陷灌进来的。
# **不许原地修**：一改 v8 的字节就变，历史 run 的 prompt hash 全部对不上
# （内部方案 V7 要求 v6/v7/v8 三者字节逐一不变）。缺陷连同本常量一起冻结，修好的版本另起 v9。
# 下面这段文本因此**保持手抄形态、不改成调 action_table()**：调了就会改字节。
_V8_ACTION_TABLE = """
## 六、状态→动作速查表（听出状态后，下一轮必须先做到对应第一条）
- 听出[不满]：先道歉并共情（为体验/给到的麻烦致歉），再给可执行方案；不争辩、不催办、不继续推销。
- 听出[急迫]：压缩表述先给结论与时间节点，句子短、节奏快；去掉铺垫与客套。
- 听出[疑问/不信]：先自证身份与来意，给出可核实的依据（单号/渠道/时限）；不催促决定。
- 听出[轻声/不便]：压低音量、缩短内容，主动提出改时间或换联系方式；不展开长谈。
- 听出[配合]：顺势推进，当场坐实关键要素（时间/金额/方式）；不拖沓。
- 未出现以上特殊状态：按正常流程推进，保持平稳。
"""
VOICE_CALL_INSTRUCTION_V8 = VOICE_CALL_INSTRUCTION_V7 + "\n" + _V8_ACTION_TABLE.strip()

# ---- v9（本轮正式口径，内部方案）----
# = 修好 variant 透传缺陷后的「v7 + 状态→动作速查表」（裁定 8 / 内部方案）。
# 与 VOICE_CALL_INSTRUCTION_V7 的差异只有三处，逐条对应 内部方案：
# ① **删 §一.3**「判断出状态后，按下方『用户状态应对策略』执行…」—— 那一节在 v9 已不再
# 渲染（domains/base.py 的 _NO_PLAYBOOK_VERSIONS），留着就是指向一个不存在的板块；
# ② 删掉一条后 §一–§五 的条目重新编号为 1–10（v7 是 1–11），文本本身逐字不动；
# ③ **新增 §六 速查表**，内容不手抄，直接调 domains.base.action_table() —— 全仓唯一定义处，
# 被测侧与裁判侧同源。5 行 5 态，**无**「未出现以上特殊状态」兜底行
# （内部方案：5 选 1 口径下留兜底行就是给模型一个垃圾桶）。
# 保留的 §一.1（AGENT_VOICE_STATE_RULE）与 §二.3（AGENT_DUAL_RESPONSE_RULE）讲的是「声音是判据」
# 和「双通道应答」，不是状态→动作映射，与 §六 不重复（内部方案 的处置表）。
# §三（状态优先于流程）是优先级声明，同样保留。
#
# 不写成「在 V7 字符串上做 replace/正则派生」的形式：v9 的 prompt 字节是本轮实验变量，
# 必须能整段人读、整段冻结；派生写法会让 V7 的任何后续改动静默改变 V9。
_VOICE_CALL_INSTRUCTION_V9_BODY = f"""
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

# §六 的标题写明「唯一动作口径」：§二.3 的 AGENT_DUAL_RESPONSE_RULE 里也带一串
# 「不满就先致歉共情再给方案；…」的**举例**（那是 v8 时代的旧措辞，粒度与 §六 不一致，
# 例如 [疑问/不信] 一处说「先自证身份」、§六 说「先说『明白您有疑虑』」）。该常量与裁判契约
# 逐字节共用，本轮不动它（内部方案 只禁改 ACTION_V9，但改 AGENT_DUAL_RESPONSE_RULE 会连带
# 改 v7 字节、破 V7），所以在 §六 标题上把优先级说死，避免模型读到两份口径不知从哪份。
# 这处残留冲突已登记进报告，移交主窗口/内部方案 决定是否在下一轮统一改词。
_V9_ACTION_TABLE = action_table(
    heading="## 六、状态→动作速查表（唯一动作口径：听出哪一种，下一轮就先做哪一行的开场，再接一句跟进）"
)

# 拼接用 "\n\n"（v8 用的是 "\n"）：§一–§五 每个 `##` 小节之间本来就空一行，v8 那种单换行
# 会让 §六 紧贴 §五 的最后一条，人读与模型读都显得格式断裂。此处按 v9 自身的一致性来。
VOICE_CALL_INSTRUCTION_V9 = _VOICE_CALL_INSTRUCTION_V9_BODY + "\n\n" + _V9_ACTION_TABLE
# v9_notable = v9 减去速查表（内部方案：tierAB 探针 LAYER2_TABLE=False 的「测内化」臂）。
# 板块侧口径与 v9 完全相同（同样不渲染 playbook），差异**只有**这一张表 —— 单变量。
VOICE_CALL_INSTRUCTION_V9_NOTABLE = _VOICE_CALL_INSTRUCTION_V9_BODY

# ---- v9_turnwise（2026-09-10）= v9 逐字节 + 末尾追加一节「轮次纪律」----
# 动机（两条都是实测现象，不是预设）：
# ① 部分全双工端点无服务端轮次边界、且**无人应答时不让出话权**，实测有相当比例的
# 回复是撞满 provider 的 30s 应答窗被截断的「真回复 + 模型自问自答演双方」
# （内部脚本 docstring 已登记该形态）；
# ② 该端点服务端会在会话活到 91–146s 时发 session.closed(backend_error) 掐断，
# 单轮说得越长、一通电话能走到的轮数越少 ⇒ 关键轮越可能落在窗口外。
# ⚠️ **不改 v9 的字节**：v9 是另外三个端点已跑完的自驱多轮批次的实验变量，原地改一个字
# 就让那批数据与新批不可比（v8 当年就是因为原地改不得才另起 v9，见 domains/base.py）。
# 追加而不插入，也是为了让 §一–§六 的编号与 v9 逐条对齐 —— 差异严格只有这一节。
# ⚠️ 用它跑出来的读数与 v9 批次**不是同一提示词条件**，跨端点比绝对值时必须显式标注；
# 同端点内部的显性/中性配对差不受影响（两个条件用的是同一份提示词）。
_V9_TURN_DISCIPLINE = """
## 七、一次只说这一轮的话
11. 你每次开口只说"现在这一轮该说的"：把这一轮的意思说完就停下来，把话头交回客户，
    等客户真的开口回应之后再往下说。
12. 不要替客户说话，也不要先假设客户会怎么答、然后接着自己演下去；不要把后面几轮
    要说的内容一次讲完。这通电话是你和客户轮流说话，不是你一个人讲到底。
13. 唯一的例外是收尾轮：确认完最后的时间/金额/方式并道别的那一轮，可以一次说完并结束通话。
""".strip()
VOICE_CALL_INSTRUCTION_V9_TURNWISE = VOICE_CALL_INSTRUCTION_V9 + "\n\n" + _V9_TURN_DISCIPLINE

_VOICE_CALL_INSTRUCTIONS = {"v6": VOICE_CALL_INSTRUCTION_V6, "v7": VOICE_CALL_INSTRUCTION_V7,
                            "v8": VOICE_CALL_INSTRUCTION_V8,
                            "v9": VOICE_CALL_INSTRUCTION_V9,
                            "v9_notable": VOICE_CALL_INSTRUCTION_V9_NOTABLE,
                            "v9_turnwise": VOICE_CALL_INSTRUCTION_V9_TURNWISE}


def voice_call_instruction(variant: str | None = None) -> str:
    """按生效的 Agent prompt 版本取通话指令原文。"""
    return _VOICE_CALL_INSTRUCTIONS[resolve_agent_prompt_version(variant)]


# 当前默认版本的通话指令（prompt 审阅页与旧引用沿用此名）。
# 2026-09-06 决策记录：`Settings.agent_prompt` 默认值由 "v7" 翻成 "v9"（runtime/config.py），
# 本别名同步指向 V9，以保持「别名 == 不传参时拿到的版本」这一不变量（内部方案 登记的不一致已闭合）。
# ⚠️ 实测本别名**无代码消费者**：全仓 `voice_call_instruction()` 不传参的调用为 0 处，
# 唯一引用是 report/export_web.py 里三处**说明文字**（1517/1570/1832 行），故翻它不改变任何行为。
# ⚠️ 重评旧 run 仍必须显式传 v7/v6/v8，否则裁判会按被测方当时没收到的规则打分。
VOICE_CALL_INSTRUCTION = VOICE_CALL_INSTRUCTION_V9

TOOL_DISABLED_OUTPUT = '{"error": "tool execution disabled in this run"}'


@dataclass
class AgentTickResult:
    """Agent 一个 tick 的输出（编排器消费）。"""

    audio: bytes = b"" # 本 tick 播出的客服音频（24k PCM16，未填充）
    transcript: str = "" # 与本 tick 播出音频成比例的转写增量
    tool_calls: list[dict[str, Any]] = field(default_factory=list) # 含 name/arguments/call_id
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    response_done: bool = False
    done_status: str = "" # response.done 携带的状态（completed/cancelled/failed/incomplete）
    speech_started: bool = False # server VAD 打断信号
    was_truncated: bool = False
    truncated_bytes: int = 0
    has_activity: bool = False
    agent_output: bool = False # 本 tick 收到下行产出事件（音频/转写/done）——供编排器判轮
    buffer_bytes: int = 0 # 适配器中尚未播出的缓冲音频
    tools_flushed: int = 0 # 本 tick 投递的工具结果数
    errors: list[str] = field(default_factory=list)


class Agent:
    """被测 Agent（音频原生 Realtime，tick 级全双工）。"""

    def __init__(
        self,
        model_spec: str,
        api_key: str,
        domain: DomainSpec,
        system_prompt: str,
        voice: str | None = None, # None = 取模型绑定表音色（registry 内解析）
        tick_ms: int = 200,
        vad_mode: str = "server_vad",
        tools_enabled: bool = False, # False（默认）：start() 不注册工具（评测不发起 function call）
        prompt_variant: str | None = None, # None = 取 Settings.agent_prompt（见 AGENT_PROMPT_VERSIONS）
    ) -> None:
        self.model_spec = model_spec
        self._api_key = api_key
        self._voice = voice # 故障转移时要按同一音色重建 provider，故必须留住
        # ⚠️ 必须留住 tick_ms：_switch_volc_key() 重建 adapter 时要用 self.tick_ms。
        # 这行缺失时，豆包**每次触发换 key 都抛 AttributeError**，并且因为它在
        # start() 的 except 分支里，原始连接错误还会被这个假错误盖掉
        # （2026-09-07 T1 批实测 289 次 " 'Agent' object has no attribute 'tick_ms' "）。
        self.tick_ms = tick_ms
        self.provider = create_provider(model_spec, api_key, voice=voice)
        self.adapter = QwenTickAdapter(self.provider, tick_ms=tick_ms)
        self.domain = domain
        self.system_prompt = system_prompt
        self.vad_mode = vad_mode
        self.tools_enabled = tools_enabled
        self.prompt_variant = resolve_agent_prompt_version(prompt_variant)
        self.domain_db = None # 工具执行预留：attach_db 注入后才启用

    # ---- 生命周期 ----
    async def start(self) -> None:
        # 开关关闭时不注册任何工具（cli probe-realtime-voices 已验证不传 tools WSS 正常）；
        # 开关打开时恢复域工具注册（可逆）。
        tools = [] if not self.tools_enabled else [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.domain.tools
        ]
        full_prompt = f"{voice_call_instruction(self.prompt_variant)}\n\n{self.system_prompt}"
        try:
            await self.adapter.connect(
                system_prompt=full_prompt,
                tools=tools,
                vad_mode=self.vad_mode,
            )
        except Exception as exc: # 只为换 key 后重试一次；换不动就原样抛，不吞错
            if not self._switch_volc_key(exc):
                raise
            await self.adapter.connect(
                system_prompt=full_prompt,
                tools=tools,
                vad_mode=self.vad_mode,
            )

    def _switch_volc_key(self, exc: BaseException) -> bool:
        """豆包（volc）多 key 故障转移：第一把额度用完/鉴权失败 ⇒ 换下一把重建 provider。

        作者 2026-09-06 裁定：「现在给了两个火山引擎 API，第一个用完了再用 2」。

        ⚠️ **刻意不按错误文案筛选**（只在"看起来像额度不足"时才切）：额度耗尽的真实报错
        文案我**无法在不烧光一把 key 的前提下实测**，靠猜关键词会出现"该切时没切"。
        池子只有 2 深 ⇒ 连接失败就换下一把、多试一次，**严格优于直接失败**；
        若是协议/网络类故障，第二把也会失败，届时两个错误一起抛出，不丢信息。
        非 volc 后端一律返回 False（不改变 qwen 链路的任何行为）。
        """
        from listen2serve.model_registry import resolve_model
        from listen2serve.runtime.config import get_settings

        try:
            backend, _name = resolve_model(self.model_spec, "")
        except Exception: # noqa: BLE001 - 解析不出来就不是 volc，走原错误
            return False
        if backend != "volc":
            return False
        s = get_settings()
        nxt = s.volc_mark_exhausted(self._api_key)
        if not nxt:
            return False
        # ⚠️ 只记 key 前 6 位用于区分是哪一把，**绝不记全 key**（本仓纪律）
        logger.warning(
            "豆包 realtime 连接失败，切换火山 API key：%s… → %s…（原错误：%s）",
            (self._api_key or "")[:6], nxt[:6], f"{type(exc).__name__}: {exc}"[:200],
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
    def session_usage(self) -> dict[str, int]:
        """本会话累计 Realtime token 用量（审计/熔断用）。"""
        return self.adapter.session_usage

    @property
    def needs_continuous_audio(self) -> bool:
        """端点是否要求上行音频流不中断（目前只有豆包全双工声明）。

        用途有两个：① provider 内部已据此开静音保活；② 编排器对这类端点放宽单轮
        等待上限 —— 2026-09-08 实测同一条 1.92s 用户音频，豆包首响在 1.7s 与 24.7s
        之间跳（重尾），而 `max_response_ticks=30s` 一到就直接判死整通（旧批 17/48 是
        这么死的）。qwen 侧不声明该属性 ⇒ 仍按 30s，**既有端点时序口径不变**。
        """
        return bool(getattr(self.provider, "needs_continuous_audio", False))

    def stream_diag(self) -> dict[str, Any]:
        """端点自定义的上行流诊断（豆包：静音保活补发帧数）；无此能力的端点返回空。"""
        fn = getattr(self.provider, "stream_diag", None)
        return dict(fn()) if callable(fn) else {}

    # ---- tick 交互 ----
    async def run_tick(self, user_pcm16: bytes, tick: int = 0) -> AgentTickResult:
        """运行一个 tick；工具调用当场执行并将结果排队至下一 tick 投递。"""
        r = await self.adapter.run_tick(user_pcm16, tick)
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
        )
        for fc in r.tool_calls:
            args = _parse_args(fc.get("arguments", "{}"))
            record = {"call_id": fc["call_id"], "name": fc["name"], "arguments": args}
            result.tool_calls.append(record) # 记录始终保留（评测/审计用）
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
