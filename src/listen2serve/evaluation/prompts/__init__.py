"""Judge prompt 模板（随评测集发布，版本固定）。

B2 起裁判 prompt 冻结版本号 + 文本 hash（F1：版本可追溯），
每条 verdict 由 cli evaluate 组装 judge_meta 写入。
"""

from __future__ import annotations

import hashlib
from typing import Any

# 内部方案：裁判侧的动作口径从 domains.base 单一来源注入（不在此处硬编码文字）。
# 依赖方向安全：domains/* 从不 import evaluation.prompts（已实测），不会成环。
from listen2serve.domains.base import judge_action_table

# ---- F1：裁判 prompt 版本追溯 ----
JUDGE_PROMPT_VERSION = "v2.5-K" # 冻结版本号（登记升版记录后方可改）；v2.5-K：内部方案 弦外之音量规
# 内部方案 升档（v2.5-K）：key_behavior_met（是/部分/否）升级为 key_turn_score（1-5 锚定式量规，
# MOS 口径）：评的是客服对用户关键轮「弦外之音」（语气表达的真实状态/真意，未必见于文字）
# 的听出与承接质量；流程性动作归 FlowRate；passed = score≥4 ∧ 禁止零触发；state_fit 保留为诊断维度。
# v2.4-K（过渡）：key_behavior_met 收敛为即时响应妥当性；v2.3-J：判据结构三项化。
# 内部方案 升档：情绪状态 11→6 态（neutral/cooperative/displeased/urgent/doubtful/
# hushed），契约状态规则面随之变化；Voice 裁判 prompt 增 6 态图例（裁判可见文案
# 变化，hash 变）。判据结构（逐动作分解/三维声音评分/对话级 0-2 分）未动。

# 内部方案（v2.3-J）：judge prompt 冻结解除（README 跨计划禁区第 3 条，升版记录已登记）。
# 关键轮裁判判据收敛（KeyTurnJudge）+ 新增全对话裁判（FlowJudge：固定清单逐条
# 命中 + 终局结果分，单次调用合并产出）；v2.2-I 三模板原文保留，供 --legacy-judges 切回。
JUDGE_PROMPT_VERSION_V23J = "v2.3-J"
# 内部方案：keyturn/flow 裁判各自独立版本追踪（keyturn 升 1-5 量规，flow 未动）
KEYTURN_JUDGE_VERSION = "v2.7-S" # 内部方案：动作口径同源化 + 判据优先级 + 无音频声明（见下方说明）
# 内部方案 升档（v2.7-S）三点变更，判据结构（1-5 量规 / forbidden_items / state_fit 诊断）**未动**：
# ① 动作速查不再硬编码文字，改为模块加载期插入 domains.base.judge_action_table()
# （= ACTION_V9 的裁判侧渲染，全仓唯一动作口径来源）。原两行压缩速查是 v8 措辞
# （「疑问/不信→先自证身份」「配合→顺势推进坐实要素」），与被测侧 v9 表（「先说明白您有
# 疑虑」「先说好的」）不同源 ⇒ 被测按一份做、裁判按另一份判（内部方案 问题 3）。
# 裁判侧只把 opening 当判据、follow 显式标注不单独扣分（ACTION_V9 的定义即如此）。
# ② 新增「判据优先级」段：速查表 + 锚点【承接】是打分的动作判据；契约其余板块
# （逐态必须/可选、通用约束里的动作举例、业务流程与收口）只作背景，不得作为加减分依据。
# 理由：契约里 AGENT_DUAL_RESPONSE_RULE 内嵌同一串 v8 措辞举例，而原文写着「以契约
# 『通用约束』的双通道应答规则为准，此处为速查」——不声明优先级，新注入的 v9 表会被
# 上一句压成附注（内部方案 §残留 1 已把这处冲突移交 内部方案）。
# ③ 新增「本轮无客服音频」声明：Layer R（C/D/B2/B3）只收文本回复、没有客服音频，
# 双通道应答规则的（二）语气转变部分不可判，明示不适用，避免裁判因一条它无法核验的
# 规则压分。语气通道在有音频的臂仍由 Voice 裁判独立评分。
# 同时删除的旧句：无（v2.6-P 的证据窗口声明、1-5 锚点、forbidden_items 口径全部逐字保留）。
KEYTURN_JUDGE_VERSION_V26P = "v2.6-P" # 冻结：2026-09-06 之前的 run 用此版本复算
FLOW_JUDGE_VERSION = "v2.3-J" # FlowJudge 判据未变，沿用 v2.3-J
# 内部方案 审阅修正：Voice 判据自 内部方案 后未再改动，但原来复用 JUDGE_PROMPT_VERSION，
# 被 内部方案 全局升版连带误标为 v2.5-K（judge_meta.voice.version 与注释矛盾，K18 审阅 P7）；
# 此处拆为独立常量，与 keyturn/flow 同构各自跟踪。
VOICE_JUDGE_VERSION = "v2.3-P" # 内部方案：state_voice_fit 改判「语气是否为听到的状态而转变」
# K18c：泄露探针（LeakageProbe）——不评客服，只盲评用户关键轮文本本身的态度表露量，
# 给 T1/T2/T3 三层透明度“是否真的分级”一个与被测模型无关的独立度量（manipulation check）。
LEAKAGE_PROBE_VERSION = "v1.0-K18c"


def prompt_hash(text: str) -> str:
    """prompt 文本 sha256 前 16 位（版本追溯用）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


POLICY_JUDGE_PROMPT = """你是一名客服策略合规裁判。请基于给定的策略契约、对话历史、用户状态真相（oracle）与客服的关键轮回复，对契约中的每条必需动作与禁止动作逐一分解判定，再汇总结论，并输出 JSON。

## 判定要求
1. required_items：对契约要求本次判定的每条必需动作逐条判定命中情况
   - hit："是"（回复中明确执行）/"否"（未执行）/"部分"（执行但不充分）
   - evidence：引用客服回复中的关键依据（无则写"无"）
2. forbidden_items：对契约的每条禁止动作逐条判定是否触发
   - triggered："是"/"否"；evidence 同上
3. state_fit：客服回复是否与用户当前状态适配？（是/否/过度/不足）
   - "过度"：反应强于状态所需（如对轻微不满使用过度安抚/对轻微抗拒过度施压）
   - "不足"：未充分回应状态（如面对强烈抗拒仍按普通流程推进）
4. context_coherent：客服回复是否与对话历史上下文连贯？（是/否）
5. required_rate：必需动作中 hit="是" 的比例（0-1，保留两位小数）

## 策略契约
{policy_rules}

## 对话历史
{history}

## 用户状态真相（oracle）
{oracle_state}

## 关键轮客服回复
{agent_reply}

## 输出格式
只输出 JSON：
{{
  "required_items": [{{"action": "必需动作原文", "hit": "是|否|部分", "evidence": "回复中的依据"}}, ...],
  "forbidden_items": [{{"action": "禁止动作原文", "triggered": "是|否", "evidence": "回复中的依据"}}, ...],
  "state_fit": "是|否|过度|不足",
  "context_coherent": "是|否",
  "required_rate": 0-1,
  "reason": "简要理由（中文，50字内）"
}}
"""

# ---- 内部方案（v2.3-J）：关键轮裁判（KeyTurnJudge） ----
# 修复 v2.2-I 结构性缺陷（planJ）：判定对象收敛为关键轮单条回复 × 单锚点，
# 删除 required_items 逐条判定（移入 FlowJudge 全对话口径），证据窗口显式声明。
# 内部方案（v2.7-S）：动作速查改为**模板槽位** + 模块加载期插入，本文件不再存第二份动作措辞。
def _indent_block(text: str, pad: str = "     ") -> str:
    """整块缩进，供插进 prompt 的编号列表项之下（空白行不加垫）。"""
    return "\n".join(pad + ln if ln.strip() else ln for ln in text.split("\n"))


# 裁判侧动作速查：由 domains.base.judge_action_table()（= ACTION_V9 的裁判侧渲染）产出。
KEYTURN_ACTION_CHEATSHEET = _indent_block(judge_action_table())

# 模板与成品分开存：正文里的 JSON 示例是 {{ }} 转义形态，供 KeyTurnJudge 后续
# .format(policy_rules=…, history=…, oracle_state=…, agent_reply=…) 使用。
# 因此速查插入必须走 str.replace 而不是 str.format —— 在此处 format 会把双花括号提前压成
# 单花括号，下游 format 直接 KeyError（这是本模块唯一的注入点，别处不要再 format 模板）。
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

# 成品 prompt（对外只暴露这一个名字；KeyTurnJudge 与 judge_meta hash 都用它）
KEYTURN_JUDGE_PROMPT = _KEYTURN_JUDGE_PROMPT_TMPL.replace(
    "{action_cheatsheet}", KEYTURN_ACTION_CHEATSHEET)
assert "{action_cheatsheet}" not in KEYTURN_JUDGE_PROMPT, "速查槽位未被替换（judge_action_table 返回空？）"
assert KEYTURN_JUDGE_PROMPT.count("{{") == _KEYTURN_JUDGE_PROMPT_TMPL.count("{{"), \
    "注入破坏了 JSON 示例的双花括号转义（下游 .format 会 KeyError）"

# ---- 内部方案（v2.3-J）：全对话裁判（FlowJudge） ----
# 单次调用合并产出：过程清单 flow_items（数据侧固定清单，裁判不得增删）+ 终局
# task_score（0-2，聚合层归一 0/0.5/1）；替代 v2.2-I 的 DIALOGUE_TASK/TRANSITION
# 两个独立裁判（完整对话正文只进一次 prompt，控 token）。
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

# ---- 截断观察窗追加段（2026-09-11，为「通话物理上跑不完」的端点而加）----
# 为什么必须另开一口径：被调端点的服务端会话只能活 91-146 s，而其余端点跑完
# 一通需 176-238 s（p50）⇒ 它的对话必然中途结束。拿中止的对话去评整通 FlowRate（清单
# p50=5 条、其中 43.9% 是收尾/确认类）与终局 task_score，量到的是「走到了第几轮」而不是
# 「能力如何」。故把四端点都截到前 K 轮，在同一观察窗内比。
# 本段**追加**在 FLOW_JUDGE_PROMPT 之后、不改原模板：三家已入文档的整通读数不得被重评改动。
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

DIALOGUE_TASK_JUDGE_PROMPT = """你是一名客服任务完成度裁判。请基于任务目标、策略规则与完整对话记录，判定客服任务完成度（0-2 分），并输出 JSON。

## 评分标准
- 2：任务目标完全达成（如：促成还款承诺/完成产品推介/解决用户问题），且无违规
- 1：任务部分推进但未完全达成（如：用户同意考虑但未给承诺），无违规
- 0：任务未推进，或触发禁止动作，或对话无效循环

## 任务目标
{task_goal}

## 策略规则
{policy_rules}

## 完整对话记录
{history}

## 输出格式
只输出 JSON：
{{
  "score": 0|1|2,
  "reason": "简要理由（中文，50字内）"
}}
"""

DIALOGUE_TRANSITION_JUDGE_PROMPT = """你是一名客服状态转变适应度裁判。请基于策略规则、完整对话记录与状态转变真相（oracle），判定客服在状态转变前后的整体表现（0-2 分），并输出 JSON。

## 评分标准
- 2：转变前策略正确 + 转变后识别了状态变化 + 调整了策略行为
- 1：部分满足：识别了变化但调整不充分，或转变前正确但转变后滞后 1-2 轮
- 0：未识别状态变化（继续旧策略），或转变后触发禁止动作

（口径说明：本维度仅基于文本策略行为判定；B2 起声音相关子项不再纳入本维度，声音表现由 Voice Judge 独立评分。）

## 状态转变
{transition_desc}

## 策略规则
{policy_rules}

## 完整对话记录
{history}

## 输出格式
只输出 JSON：
{{
  "score": 0|1|2,
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

# ---- F1：各 prompt 文本 hash（模块加载即冻结）----
POLICY_JUDGE_PROMPT_HASH = prompt_hash(POLICY_JUDGE_PROMPT)
VOICE_JUDGE_PROMPT_HASH = prompt_hash(VOICE_JUDGE_PROMPT)
DIALOGUE_TASK_JUDGE_PROMPT_HASH = prompt_hash(DIALOGUE_TASK_JUDGE_PROMPT)
DIALOGUE_TRANSITION_JUDGE_PROMPT_HASH = prompt_hash(DIALOGUE_TRANSITION_JUDGE_PROMPT)
# 内部方案（v2.3-J）新模板 hash
KEYTURN_JUDGE_PROMPT_HASH = prompt_hash(KEYTURN_JUDGE_PROMPT)
# 内部方案（v2.7-S）：速查表由 domains.base 注入 ⇒ 单独留一份 hash，
# 便于审计「prompt 变了是因为判据文字变了，还是因为动作表变了」。
KEYTURN_ACTION_CHEATSHEET_HASH = prompt_hash(KEYTURN_ACTION_CHEATSHEET)
FLOW_JUDGE_PROMPT_HASH = prompt_hash(FLOW_JUDGE_PROMPT)
# K18c 新增：泄露探针（操纵有效性检验，与客服表现无关）
LEAKAGE_PROBE_PROMPT_HASH = prompt_hash(LEAKAGE_PROBE_PROMPT)


def build_judge_meta(
    policy_model: str = "",
    voice_model: str = "",
    dialogue_model: str = "",
    dialogue_metric: str | None = None,
    profile_injected: bool | None = None,
    judge_schema: str = "v2.2-I",
) -> dict[str, Any]:
    """组装 verdict 级 judge_meta（F1）：policy/voice/dialogue 各自的版本、hash 与裁判模型。

    profile_injected（v5.9 契约内容维度标记）：policy/dialogue 裁判契约输入末尾是否
    追加了同源客户信息档案（render_customer_profile，v5.8 起；仅元信息，不改判据与
    prompt 模板）。v5.8 之前产出的 verdict 无此键 → 可按键缺失/取值程序化甄别
    v5.8 断点前后的判定（裁判可见事实面不同，分数不宜混比）；Voice 裁判不注入档案，
    故 voice 条目不携带该标记。

    judge_schema："v2.2-I"（legacy：policy+dialogue 双裁判）或
    "v2.3-J"（keyturn+flow 双裁判；voice 不变）。两种 schema 的 meta 结构不同，
    续评指纹天然互斥（不同 schema 的 partial 不可混用）。
    """
    if judge_schema == "v2.3-J":
        meta: dict[str, Any] = {
            "schema": "v2.3-J",
            "keyturn": {
                "version": KEYTURN_JUDGE_VERSION,
                "prompt_hash": KEYTURN_JUDGE_PROMPT_HASH,
                # 内部方案（v2.7-S）：动作口径由 domains.base 注入，单独入指纹。
                # 动作表若变（= 尺子变），即使 prompt 模板一字未动也必须让旧 partial 失效。
                "action_cheatsheet_hash": KEYTURN_ACTION_CHEATSHEET_HASH,
                "model": policy_model,
            },
            "flow": {
                "version": FLOW_JUDGE_VERSION,
                "prompt_hash": FLOW_JUDGE_PROMPT_HASH,
                "model": dialogue_model,
            },
            "voice": {
                "version": VOICE_JUDGE_VERSION, # Voice 判据未动，独立跟踪（不随全局版本号漂移）
                "prompt_hash": VOICE_JUDGE_PROMPT_HASH,
                "model": voice_model,
            },
            # K18c：泄露探针亦入指纹——新增该探针后 verdict 多了 leakage_* 键，
            # 旧 partial 必须失效重评，否则会被错误复用成“指标缺失”的半成品。
            "leakage": {
                "version": LEAKAGE_PROBE_VERSION,
                "prompt_hash": LEAKAGE_PROBE_PROMPT_HASH,
                "model": policy_model,
            },
        }
        if profile_injected is not None:
            meta["keyturn"]["profile_injected"] = profile_injected
            meta["flow"]["profile_injected"] = profile_injected
        # 裁判契约（render_judge_contract）随 Agent prompt 版本变化 → 入续评指纹，
        # 否则 v6 run 的 partial 会被 v7 契约的重评静默复用。
        from listen2serve.domains.base import resolve_agent_prompt_version

        meta["agent_prompt"] = resolve_agent_prompt_version()
        return meta
    meta: dict[str, Any] = {
        "policy": {
            "version": JUDGE_PROMPT_VERSION,
            "prompt_hash": POLICY_JUDGE_PROMPT_HASH,
            "model": policy_model,
        },
        "voice": {
            "version": VOICE_JUDGE_VERSION,
            "prompt_hash": VOICE_JUDGE_PROMPT_HASH,
            "model": voice_model,
        },
    }
    if profile_injected is not None:
        meta["policy"]["profile_injected"] = profile_injected
    if dialogue_metric == "transition_score":
        meta["dialogue"] = {
            "version": JUDGE_PROMPT_VERSION,
            "prompt_hash": DIALOGUE_TRANSITION_JUDGE_PROMPT_HASH,
            "model": dialogue_model,
        }
    elif dialogue_metric == "task_completion":
        meta["dialogue"] = {
            "version": JUDGE_PROMPT_VERSION,
            "prompt_hash": DIALOGUE_TASK_JUDGE_PROMPT_HASH,
            "model": dialogue_model,
        }
    else:
        meta["dialogue"] = None
    if meta["dialogue"] is not None and profile_injected is not None:
        meta["dialogue"]["profile_injected"] = profile_injected
    return meta
