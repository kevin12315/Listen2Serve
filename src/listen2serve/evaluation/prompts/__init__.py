"""Published multi-turn judge prompts and content fingerprints."""

from __future__ import annotations

import hashlib
from typing import Any

from listen2serve.domains.base import judge_action_table

JUDGE_PROMPT_VERSION = "1.0"
KEYTURN_JUDGE_VERSION = JUDGE_PROMPT_VERSION
FLOW_JUDGE_VERSION = JUDGE_PROMPT_VERSION
VOICE_JUDGE_VERSION = JUDGE_PROMPT_VERSION
LEAKAGE_PROBE_VERSION = JUDGE_PROMPT_VERSION


def prompt_hash(text: str) -> str:
    """Return a compact SHA-256 fingerprint of a prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _indent_block(text: str, pad: str = "     ") -> str:
    """整块缩进，供插进 prompt 的编号列表项之下（空白行不加垫）。"""
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.split("\n"))


KEYTURN_ACTION_CHEATSHEET = _indent_block(judge_action_table())

_KEYTURN_JUDGE_PROMPT_TMPL = """你是一名客服关键轮应对裁判。判定对象仅为**关键轮的一条客服回复**。用户的关键轮台词往往是“字面一个意思、语气另一个意思”（弦外之音）：文字未必表露真实状态，语气才承载真意。请基于策略契约（重点看「关键事件应对锚点」中的【表面】【弦外】【承接】三要素）、对话历史（仅供理解上下文）、用户状态真相（oracle）与待判定回复，完成三项判定并输出 JSON。

## 证据窗口声明（必须遵守）
- 判定对象仅为下方「关键轮客服回复」这一条发言；
- 对话历史仅用于理解上下文（如判断前轮是否已提供某信息），**不得**作为评分或 forbidden_items 的证据；
- evidence 一律引用本轮回复原文（无则写“无”）。

## 判据（三项，职责分离，不重复计分）
1. key_turn_score：客服是否听出用户关键轮的弦外之音，并在**文本策略**上做出该状态对应的动作？（1-5 整数，锚定式量规）
   - 本维度只评文本通道。你看不到音频，**不要推测客服的语气**；语气通道在有客服音频的实验臂由 Voice 裁判独立评分。
   - 本轮待判回复**没有音频**：契约「通用约束」中双通道应答规则的（二）语气转变部分**不适用于本判定**，
     不得因“声音没跟着变”扣分，也不得因“读起来够软”加分。
{action_cheatsheet}
   - 判据优先级（必须遵守）：上面这张表 + 契约「关键事件应对锚点」的【承接】= 本轮打分的动作判据。
     契约其余板块（「状态应对规则」的逐态必须/可选、「通用约束」里的动作举例、「业务流程与收口」）
     只作背景理解，**不得**作为 key_turn_score 的加分或扣分依据；
     唯一例外是「场景动作约束」里的**禁止动作** —— 它是下面第 2 项 forbidden_items 的判定来源。
   - 5 = 准确听出弦外之音，文本策略正是该状态所需且分寸恰当（情绪/真意被回应，事情也在推进，不过度）
   - 4 = 听出并采取了方向正确的文本策略，分寸略欠（安抚稍弱/推进稍急，不影响大局）
   - 3 = 接住部分信号（有口头安抚或试探）但文本策略仍偏字面
   - 2 = 只回应字面文本，措辞略软，无该状态对应的策略动作
   - 1 = 完全未接住，按字面硬推（继续推销/催逼/空泛承诺）
   - 评分要点：锚点中的流程动作（给方案/促承诺/收口）**不作为本轮判据**（归 FlowRate 全对话口径）；
     用户当轮台词未给出某动作的触发条件时，该动作不得作为扣分依据；
     双向动态看状态转变苗头的察觉与承接，全正/全负动态看情绪外显动作的接住
2. forbidden_items：仅对契约「场景动作约束」板块的**禁止动作**逐条判定本轮是否触发
   - triggered：“是”/“否”；evidence 同上（仅看本轮）
   - 注意：「通用约束」中的字数/格式类约束（如「回复控制在 50 字以内」）**不计**禁止触发；
     复读机式重复也不计；本维度只判行为类禁止（威胁/过度承诺/推诿/施压等）
3. state_fit：回复**措辞的力度**是否适配用户当前状态？（是/过度/不足，诊断维度；仍只看文本）
   - “过度”：措辞力度强于状态所需（如对轻微不满连篇致歉安抚）
   - “不足”：措辞力度弱于状态所需（如面对强烈抗拒仍平铺直叙按流程推进）

## 策略契约
{policy_rules}

## 对话历史（仅上下文，非证据）
{history}

## 用户状态真相（oracle）
{oracle_state}

## 关键轮客服回复（判定对象）
{agent_reply}

## 输出格式
只输出 JSON：
{{
  "key_turn_score": 1-5,
  "key_turn_evidence": "本轮回复中承接弦外之音的依据（无则写无）",
  "forbidden_items": [{{"action": "禁止动作原文", "triggered": "是|否", "evidence": "本轮回复中的依据"}}, ...],
  "state_fit": "是|过度|不足",
  "reason": "简要理由（中文，50字内）"
}}
"""

KEYTURN_JUDGE_PROMPT = _KEYTURN_JUDGE_PROMPT_TMPL.replace("{action_cheatsheet}", KEYTURN_ACTION_CHEATSHEET)

FLOW_JUDGE_PROMPT = """你是一名客服全对话裁判。请基于策略契约、完整对话记录与给定的过程清单，一次完成两项判定：过程清单逐条命中（flow_items）与终局结果分（task_score），并输出 JSON。

## 证据窗口声明
- 任意轮次的客服发言均可作为证据，每条必须标注命中轮次（对话记录已标注轮号）；
- 过程清单为固定清单：不得增删、改写条目，逐条按原文判定。

## 一、过程清单（flow_items）
对下列每条动作判定：
- hit："是"（全程某轮明确执行）/"部分"（有执行但不充分）/"否"（全程未执行）
- turn：命中所在轮次（hit="否" 时为 null）
- evidence：客服发言依据（引用原文，无则写"无"）

命中口径（重要）：
- **隐式执行同样计命中**：确认类动作（如确认问题解决/确认意向）可由用户后续表态佐证
  （用户接受方案/道谢道别/顺着推进而无异议，即视为该确认动作已生效）；
- 动作在全程任意轮执行即计命中，不要求集中在某一轮；
- 判定宁宽勿苛：只要对话中有实质执行证据即计"是"，仅措辞不完美不降"部分"。

输出要求：flow_items **严格按清单顺序逐条输出，action 字段与清单原文逐字一致**，
不得改写、合并或重新表述。

清单：
{checklist}

## 二、终局结果分（task_score，0-2；只看终局结果，不因过程细节扣分）
- 2：任务目标完全达成（承诺锁定/问题解决/转化或预约达成等）；用户状态发生转变的场景，转变后仍达成目标或合理调整目标并达成，同样计达成
- 1：任务部分推进但未完全达成（如用户同意考虑但未给承诺）
- 0：任务未推进，或对话无效循环

## 任务目标
{task_goal}

## 策略契约
{policy_rules}

## 完整对话记录（已标注轮号）
{history}

## 输出格式
只输出 JSON：
{{
  "flow_items": [{{"action": "清单动作原文", "hit": "是|部分|否", "turn": 命中轮次或null, "evidence": "客服发言依据"}}, ...],
  "task_score": 0|1|2,
  "reason": "简要理由（中文，50字内）"
}}
"""

FLOW_JUDGE_WINDOW_SUFFIX = """
## 附加要求：本次只给到前 {turn_cap} 轮（截断观察窗）
本次提交的对话记录**只包含前 {turn_cap} 轮**（不是整通）。请在此窗口内判定，并遵守：
- 每条动作额外输出 `in_window`：按策略契约与正常推进节奏，该动作**到第 {turn_cap} 轮为止是否已到应当执行的时点**。
  典型地，开场/报身份/说清来意属早期；方案与金额确认属中期；收口、约定时间、道别属末尾几轮。
- 本应发生在窗口之后的动作：`in_window` 记 false，`hit` 记 "否"（它只是没轮到做，不是做得差）；
- 已在窗口内执行的动作（hit="是"/"部分"）：`in_window` **必须**为 true；
- 本次**不给终局结果分**：`task_score` 固定输出 null（对话未结束，终局无从判定）。
"""

VOICE_JUDGE_PROMPT = """你是一名客服语音质量裁判。请直接聆听客服的关键轮回复音频，结合角色设定、声音要求、用户状态与回复文本，对以下三个维度分别打 1-5 分（整数），并输出 JSON。

## 评分维度（含锚点）
1. 角色声音匹配：音色/语气/语速是否符合下方「角色设定」与「声音要求」。
   - 5 = 完全符合声音要求（语气、语速、力度均贴合角色与要求描述）
   - 3 = 基本符合，但个别方面偏离要求（如偏软/偏快/缺乏角色感）
   - 1 = 与声音要求明显不符
2. 状态声音适配：客服的**语气是否随他听到的用户状态发生了应有的转变**？（本次评测的重心维度）
   - 各状态所需的语气转变：不满→放缓语速、压低音量、语气放软；急迫→加快节奏、句子变短、去掉铺垫；
     疑问/不信→放慢、加重关键信息、语气笃定；轻声/不便→压低音量、简短收束；配合→保持平稳积极。
   - 5 = 语气明确发生了该状态所需的转变，力度与节奏都到位
   - 4 = 转变方向正确，力度略欠
   - 3 = 方向不错但几乎听不出转变（与平铺直叙的念稿语气差别很小）
   - 2 = 语气基本没有变化，只是话里的内容在应对
   - 1 = 语气与用户状态相悖（如面对挫败用户语气轻快敷衍）
   - 注意：只评「有没有为这个状态而变」，不要因为音色本身好听/专业而加分（那是维度 1）。
3. 自然度：是否像真人？有无机械感、朗读感？
   - 5 = 接近真人客服，停顿与语调自然
   - 3 = 略有机械感但不影响听感
   - 1 = 明显机械/朗读感

## 角色设定
{role_description}
沟通风格：{communication_style}

## 声音要求（角色声音匹配维度的判定依据）
{voice_requirements}

## 用户当前状态
{user_state}
（状态码说明：neutral=中立；cooperative=配合；displeased=不满；urgent=急迫；doubtful=疑问/不信；hushed=轻声/不便）

## 客服回复文本（供参考，评分以音频为准）
{agent_text}

## 输出格式
只输出 JSON：
{{
  "role_voice_match": 1-5,
  "state_voice_fit": 1-5,
  "naturalness": 1-5,
  "reason": "简要理由（中文，50字内）"
}}
"""

LEAKAGE_PROBE_PROMPT = """你是一名只能看到文字记录的分析员。你听不到录音，拿不到说话人的语气、语速、音量，也不知道他真实的意图。请仅凭下面这两句文字作判断。

## 文字记录
客服：{agent_text}
用户：{user_text}

## 判定要求
1. surface_score：只读用户这句文字（不听录音），他的情绪或立场在字面上表露到什么程度？
   - 1＝完全看不出，就是一句中性的业务话（问细节、问流程、说事实），既无情绪词也无态度结论
   - 2＝只有一点口风（如“那个…”“能不能…”这类进退词、半句话），具体态度仍看不准
   - 3＝能看出有保留或有倒向，但态度没说完、留了口子
   - 4＝态度结论已经写在字面上（如明确说不接受/不需要/等不起/愿意配合），但没带情绪词
   - 5＝态度结论和情绪词都写在字面上（如“烦”“气”“怨”“感谢”“这事儿真叫人…”）
2. state_guess：仅凭文字猜用户此刻的状态，只能选一个：
   neutral（中性）/ cooperative（配合）/ displeased（不满）/ urgent（着急）/ doubtful（质疑）/ hushed（压低、回避）
   看不出来就选 neutral，不要猜。
3. cue：文字里让你这么判的词或短语（直接摘引原文）；若根本无依据就写“无”。

不要联想语气、不要推测弦外之音。你只评“字面上写了多少”。

## 输出格式
只输出 JSON：
{{
  "surface_score": 1|2|3|4|5,
  "state_guess": "neutral|cooperative|displeased|urgent|doubtful|hushed",
  "cue": "摘引原文或“无”"
}}
"""

KEYTURN_JUDGE_PROMPT_HASH = prompt_hash(KEYTURN_JUDGE_PROMPT)
KEYTURN_ACTION_CHEATSHEET_HASH = prompt_hash(KEYTURN_ACTION_CHEATSHEET)
FLOW_JUDGE_PROMPT_HASH = prompt_hash(FLOW_JUDGE_PROMPT)
VOICE_JUDGE_PROMPT_HASH = prompt_hash(VOICE_JUDGE_PROMPT)
LEAKAGE_PROBE_PROMPT_HASH = prompt_hash(LEAKAGE_PROBE_PROMPT)


def build_judge_meta(
    policy_model: str = "",
    voice_model: str = "",
    profile_injected: bool | None = None,
) -> dict[str, Any]:
    """Identify the published judge protocol, prompt contents and selected models."""
    meta = {
        "protocol": "multi-turn/1",
        "keyturn": {
            "version": KEYTURN_JUDGE_VERSION,
            "prompt_hash": KEYTURN_JUDGE_PROMPT_HASH,
            "action_cheatsheet_hash": KEYTURN_ACTION_CHEATSHEET_HASH,
            "model": policy_model,
        },
        "flow": {
            "version": FLOW_JUDGE_VERSION,
            "prompt_hash": FLOW_JUDGE_PROMPT_HASH,
            "model": policy_model,
        },
        "voice": {
            "version": VOICE_JUDGE_VERSION,
            "prompt_hash": VOICE_JUDGE_PROMPT_HASH,
            "model": voice_model,
        },
        "leakage": {
            "version": LEAKAGE_PROBE_VERSION,
            "prompt_hash": LEAKAGE_PROBE_PROMPT_HASH,
            "model": policy_model,
        },
    }
    if profile_injected is not None:
        meta["keyturn"]["profile_injected"] = profile_injected
        meta["flow"]["profile_injected"] = profile_injected
    return meta
