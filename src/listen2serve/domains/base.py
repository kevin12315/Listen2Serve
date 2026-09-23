"""域基类：policy 渲染、工具定义、DB schema、任务装载。

每个域（collection/marketing/hotline）实现 `build_domain()` 返回 DomainSpec，
评测运行时按 DomainSpec 装配工具与数据库。
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---- 用户状态词表（canonical 英文码 → 中文显示名，单一来源）----
# 内部方案（v6.0）：情绪状态 11 态精简为 6 态——客服可感知、可执行差异化策略的最小集；
# 与场景数据 oracle_state/key_event.state 使用的英文码一一对应（emotion.STATE_EMOTION 同码）。
# 后续批次复用一律 import 本表，勿在别处复制。
STATE_LABELS: dict[str, str] = {
    "neutral": "中立",
    "cooperative": "配合",
    "displeased": "不满",
    "urgent": "急迫",
    "doubtful": "疑问/不信",
    "hushed": "轻声/不便",
}

# ---- 旧 11 态 → 新 6 态映射（内部方案；旧 run 重评兼容）----
# N5 抗拒并入 displeased（业务立场移入 mission/user_goal）；N3 困惑并入 doubtful；
# 正向四态合并为 cooperative（感激/兴趣/放松对客服策略无差异）。
LEGACY_STATE_MAP: dict[str, str] = {
    "N0_neutral": "neutral",
    "N1_frustrated": "displeased",
    "N2_urgent": "urgent",
    "N3_confused": "doubtful",
    "N4_privacy": "hushed",
    "N5_resistant": "displeased",
    "N6_suspicious": "doubtful",
    "P1_cooperative": "cooperative",
    "P2_relaxed": "cooperative",
    "P3_grateful": "cooperative",
    "P4_interested": "cooperative",
}

# 旧 11 态中文标签（仅归一化反查用，勿作展示来源）
_LEGACY_STATE_LABELS: dict[str, str] = {
    "N0_neutral": "N0 中立/谨慎",
    "N1_frustrated": "N1 挫败/升级",
    "N2_urgent": "N2 急迫/不便",
    "N3_confused": "N3 犹豫/困惑",
    "N4_privacy": "N4 隐私受限",
    "N5_resistant": "N5 厌烦/抗拒",
    "N6_suspicious": "N6 怀疑/不信任",
    "P1_cooperative": "P1 配合/信任",
    "P2_relaxed": "P2 轻松/随和",
    "P3_grateful": "P3 感激/满意",
    "P4_interested": "P4 好奇/感兴趣",
}

# 旧码前缀 → 新码（变体标签兕底，如「N5 厌烦/敷衍」；后两位数字不区分后缀）
_LEGACY_PREFIX_MAP: dict[str, str] = {
    "N0": "neutral", "N1": "displeased", "N2": "urgent", "N3": "doubtful",
    "N4": "hushed", "N5": "displeased", "N6": "doubtful",
    "P1": "cooperative", "P2": "cooperative", "P3": "cooperative", "P4": "cooperative",
}


def normalize_state(key: str | None) -> str | None:
    """状态键归一化为新 6 态 canonical 英文码。

    兼容输入：新码精确命中 → 旧 11 态码（LEGACY_STATE_MAP）→ 旧中文标签反查
    （容忍后缀差异，如「N5 厌烦/抗拒（软拒绝）」）→ 新中文标签反查 → 句中嵌入的
    旧码（如 oracle_state 自由文本「用户处于P2_relaxed…」）。无法识别返回 None。
    """
    if not key:
        return None
    k = str(key).strip()
    if k in STATE_LABELS:
        return k
    if k in LEGACY_STATE_MAP:
        return LEGACY_STATE_MAP[k]
    for code, label in _LEGACY_STATE_LABELS.items():
        if k == label or label in k or k in label:
            return LEGACY_STATE_MAP[code]
    for code, label in STATE_LABELS.items():
        if k == label or label in k or k in label:
            return code
    m = re.search(r"([NP]\d)_\w+", k)
    if m:
        prefix = m.group(1)
        for code, new_code in LEGACY_STATE_MAP.items():
            if code.startswith(prefix + "_"):
                return new_code
    # 变体标签兕底：裸前缀码（如「N5 厌烦/敷衍」）按 _LEGACY_PREFIX_MAP 归一
    m = re.search(r"([NP]\d)(?!\d)", k)
    if m and m.group(1) in _LEGACY_PREFIX_MAP:
        return _LEGACY_PREFIX_MAP[m.group(1)]
    return None


# ---- B6：agent_policy 新可选键渲染（按字段存在性渐进渲染，无新键输出逐字节不变）----
# 顺序即渲染顺序约定（与 policy_templates.POLICY_SECTIONS 同源）。
POLICY_SECTION_KEYS = ("business_flow", "red_lines", "knowledge_base", "success_criteria", "speech_style")

# ---- Agent prompt 版本（v9：本轮正式口径 = v7 精简板块 + 同源速查表）----
# v6 = 内部方案6 沿用的文本客服策略口径（业务流程逐阶段全展开、状态触发条件不指明线索来源）。
# v7 = 上一版正式任务口径：状态判据显式声明为「声音」、状态应对优先于流程推进、强制收尾轮，
# 并把继承自文本客服策略的冗余描述压缩（业务流程折为一行一阶段、剔除与状态表重复的
# 「用户类型应对要点」知识条、剔除与状态优先相矛盾的「严格按流程顺序推进」红线）。
# v8 = ⚠️ **含 variant 透传缺陷，仅供历史 run 复算，禁止用于新实验**。
# 设计意图是「v7 + 状态→动作速查表，其余同 v7」，实际不是：当年 render_policy_sections
# 的分支判断写成 `== "v7"`，v8 落不进精简分支、掉进 v6 全量分支，把 v7 有意剔除的东西
# 又灌了回来（业务流程六阶段展开、「## 说话风格」整节、流程顺序红线、「用户类型应对要点」
# 知识条）。样例场景 MAR-ALL-03-T3 的 5016 字里，速查表只占 281 字，其余 1325 字是缺陷。
# **不许原地修**：一改 v8 的字节就变，历史 run 的 prompt hash 全部对不上（内部方案 V7）。
# 故缺陷连同 v8 一起冻结在 _LITE_POLICY_VERSIONS 白名单之外，修好的版本另起 v9。
# v9 = 本轮正式口径（裁定 8 / 内部方案「客服侧从 v8 退回 v7 + 只补速查表」）：
# ① 走 v7 的精简板块分支（缺陷已修）；
# ② 追加与裁判侧**同源 import** 的「状态→动作速查表」（ACTION_V9，5 态、无兜底行）；
# ③ 不再渲染场景 state_playbook（「# 用户状态应对策略」整节）—— 三份状态规范收成一份；
# ④ 剔除裁判不判的字数约束，并与红线去重（见 _V9_DROP_CONSTRAINT_SUBSTR /
# _V9_CONSTRAINT_COVERED_BY_RED_LINE）。
# v9_notable = v9 减去速查表（内部方案：tierAB 探针 LAYER2_TABLE=False 的「测内化」臂）。
# 与 v9 的唯一差异在 runtime/agent.py 的通话指令；板块渲染口径与 v9 完全相同。
# v9_turnwise = v9 逐字节 + 末尾追加一节「轮次纪律」（2026-09-10，措辞见 runtime/agent.py）。
# 为无服务端轮次边界的端点而加：该类端点常把后几轮一口气说完（含
# 自问自答演双方），而它的会话又只能活 91–146s ⇒ 说得越长、走到的轮数越少。
# 板块渲染口径与 v9 完全相同（同在下面两个白名单里），差异只在通话指令的那一节。
# 旧 run 重评须显式回到旧版本：`AGENT_PROMPT=v6` 或 CLI `--agent-prompt v6`。
AGENT_PROMPT_VERSIONS = ("v6", "v7", "v8", "v9", "v9_notable", "v9_turnwise")

# 走「精简板块」分支的版本白名单。**显式列出、故意不含 v8** —— v8 继续掉进 v6 全量分支
# 是有意的（见上：冻结缺陷以保历史 hash 可复算）。改这个白名单等于改实验口径，须走裁定。
_LITE_POLICY_VERSIONS = ("v7", "v9", "v9_notable", "v9_turnwise")

# 不再渲染场景 state_playbook 的版本（内部方案：状态→动作口径统一到速查表）。
# v6/v7/v8 仍渲染，保其字节不变；场景数据的 policy_rules 字段一律不动。
_NO_PLAYBOOK_VERSIONS = ("v9", "v9_notable", "v9_turnwise")

# v9 通用约束剔除项。两条纪律：
# ① 键一律用**模板稳定串**，只在渲染期生效，不动场景/域数据（照抄 _V7_DROP_* 的既有纪律）；
# ② **不用模糊相似度**（不可复现）—— 重复对照表逐条人工读过后写死（内部方案 红线 1）。
# 完整对照表由 `tests/test_agent_prompt_v9.py` 逐条锁住。
#
# 无条件剔除：字数/格式类约束。裁判侧明写不计禁止触发
# （evaluation/prompts/__init__.py:107「『通用约束』中的字数/格式类约束（如「回复控制在 50 字
# 以内」）**不计**禁止触发」）—— prompt 提要求、裁判不判，纯占篇幅，还会诱导模型为压字数
# 牺牲状态应对。
_V9_DROP_CONSTRAINT_SUBSTR = "回复控制在"
#
# 条件剔除：与「## 红线（一票否决）」重复的通用约束，只保留红线那份（红线更严/更具体）。
# 结构 = role → {通用约束原文 → 红线稳定子串（任一命中即视为已覆盖）}。
# **必须带条件**：3 个域的红线是场景数据、逐组措辞不同，实测有两组不覆盖 ——
# collection 的「调解中心」组 6 条无「据实/给定口径」红线（COL-ALL-05、COL-ALL-21）；
# hotline 的「司机」组 12 条 RL3 只讲不嘲讽不回怼、不含敷衍推诿（HOT-ALL-04、HOT-POS-27、
# HOT-POS-58）。这些场景**保留**该条通用约束，不能因为别的场景有红线就一并删掉。
# 逐条人工比对后判定为「措辞差异有意义、两份都留」的，不进本表（见报告）：
# collection「不得使用'一定''马上'等过度承诺词」（红线管权限，本条管用词）
# marketing 「不得使用'一定''保证'等绝对化用语」（同上）
# marketing 「用户表示不方便/不感兴趣时，应给出礼貌退路」（红线是禁止面，本条是必须动作）
# hotline 「不得使用'一定''马上''全力解决'等过度承诺词」（三域用词类口径一致保留）
# hotline 「不得在回复中传达'系统误判/平台出错'等责任定性」（无红线覆盖）
# hotline 「不得建议用户寻求外部渠道（除非政策允许）」（无红线覆盖）
_V9_CONSTRAINT_COVERED_BY_RED_LINE: dict[str, dict[str, tuple[str, ...]]] = {
    "collection": {
        "不得威胁、辱骂或使用恐吓性语言": ("威胁恐吓", "辱骂"),
        "不得透露欠款信息给第三方": ("透露",),
        "不得复读机式回复（任意连续10字不得与历史完全一致）": ("机械重复",),
        "逾期后果须客观陈述（如征信影响），非威胁": ("据实口径", "给定口径"),
        "不得承诺减免或延期（无权限）": ("越权承诺",),
    },
    "marketing": {
        "不得制造虚假紧迫感（'仅剩最后X个名额'除非属实）": ("虚假紧迫", "虚构紧迫"),
        "不得过度承诺效果": ("权益", "承诺未授权", "不承诺案件结果"),
        "不得在用户明确拒绝后继续施压推销": ("明确拒绝",),
        "不得复读机式回复": ("机械重复",),
    },
    "hotline": {
        "不得敷衍、推诿（'这不归我管'）": ("敷衍", "推诿"),
        "不得复读机式回复": ("机械重复",),
    },
}


def _v9_filter_constraints(
    role: str,
    constraints: list[str],
    red_lines: list[str] | None,
) -> list[str]:
    """v9 渲染期过滤通用约束：先无条件剔字数类，再剔与本场景红线重复的条目。

    被测 prompt（render_policy_prompt）与裁判契约（render_judge_contract）**共用本函数**，
    两侧同时瘦身，不制造新的「被测按 A 做、裁判按 B 判」。只对 v9/v9_notable 调用；
    v6/v7/v8 一律不走这里，字节不变。
    """
    covered = _V9_CONSTRAINT_COVERED_BY_RED_LINE.get(role, {})
    rl = red_lines or []
    out: list[str] = []
    for c in constraints:
        if _V9_DROP_CONSTRAINT_SUBSTR in c:
            continue
        keys = covered.get(c)
        if keys and any(k in r for r in rl for k in keys):
            continue
        out.append(c)
    return out


# v7 三条核心规则（单一来源）：被测 prompt 与裁判契约逐字节复用同一文本。
AGENT_VOICE_STATE_RULE = (
    "客户的真实状态常常不写在字面上，而在声音里（语速快慢、音量高低、语气松紧、停顿与犹豫、"
    "句尾上扬或收紧、是否压低嗓音）；判断客户状态以听到的声音为主要依据，字面措辞只是其中一条线索。"
)
AGENT_STATE_FIRST_RULE = (
    "业务流程的下一步与客户当前状态的要求冲突时，以状态应对为先：先接住状态、调整动作，"
    "流程留到后续轮次回补；流程可跳转、可回补，不必逐阶段走完。"
)
AGENT_CLOSING_RULE = (
    "识别到客户想结束通话（明说要挂断/说没别的事了，或声音上只剩极简应答、语气收紧、"
    "明显不愿再谈）时，必须把下一轮作为收尾轮：只做两件事——一句话复述本通已达成的结论或约定，"
    "然后道别；收尾轮不得提出新问题、不得追加推销或劝说、不得重复确认已确认过的信息，"
    "即使流程尚未走完也照此收尾。"
)
# 双通道应答：本项目考察的「听→说」不只是选对话术，还要求把状态判断落到自己的声音上。
# 两条通道必须同时出现，且必须指向同一状态——只改文本不改语气，或只改语气不改文本，都算没做到。
AGENT_DUAL_RESPONSE_RULE = (
    "听出客户状态后，你的回应必须同时体现在两个通道上，两者指向同一状态：\n"
    "（一）**文本策略**——说出的内容要针对该状态：不满就先致歉共情再给方案；急迫就压缩表述先给结论；"
    "疑问/不信就先自证身份、给可核实依据；轻声/不便就缩短内容、主动改约；配合就顺势推进坐实要素。\n"
    "（二）**语气转变**——你说话的声音也要跟着变：客户不满时放缓语速、压低音量、把语气放软；"
    "客户急迫时加快节奏、句子变短、去掉铺垫；客户疑问时放慢、加重关键信息、语气笃定；"
    "客户轻声不便时压低音量、简短收束；客户配合时保持平稳积极。\n"
    "只换措辞而声音一成不变，或只改语气而话术照旧，都算没有做到。"
)

# ---- 「状态→动作」速查表：全仓唯一定义处----
# 原本住在 内部脚本。搬来的理由：被测客服 prompt（runtime/agent.py）与
# 裁判 prompt必须吃**同一份**动作口径，否则「被测按 playbook 做、裁判按 ACTION_V9
# 判」就是系统性错判来源（内部方案 举的 [配合] 一格：playbook 要求「促成转化」，
# ACTION_V9 只要求「说好的 + 推下一步」）。
# 规范位置写死在 domains/base.py —— agent.py 本来就 import 本模块，不引入新的依赖方向；
# 内部脚本 改为 re-export，不留第二份定义（内部方案 V2）。
#
# opening = 该态的标志性开场动作（裁判的唯一判据）
# follow = 开场之后的一句跟进（写进 prompt 让回复完整，但**不参与判定**）
# 五行文本本轮**不改词**（内部方案；改词要走裁定 8 修订），逐字搬自 内部脚本。
ACTION_V9: dict[str, dict[str, str]] = {
    "displeased": {
        "opening": "先道歉，把「抱歉」或「对不起」的意思说出来",
        "follow": "再给一句可执行的下一步",
    },
    "urgent": {
        "opening": "先说「理解您比较着急」的意思",
        "follow": "再直接给结论或时间点",
    },
    "doubtful": {
        "opening": "先说「明白您有疑虑」的意思",
        "follow": "再给一条可核对的依据",
    },
    "hushed": {
        "opening": "先说「您现在不方便的话」的意思",
        "follow": "再提出改个时间或换个方式联系",
    },
    "cooperative": {
        "opening": "先说「好的」的意思",
        "follow": "再一句把事情推到下一步",
    },
}
# 表内行序固定，避免因顺序变化引入位置效应（原 内部脚本）
ACTION_TABLE_ORDER = ("displeased", "urgent", "doubtful", "hushed", "cooperative")
# 裁判侧的一级标题（搬家前 action_table() 的历史输出，re-export 后须逐字节不变）
ACTION_TABLE_HEADING = "# 语气应对规范（听出哪一种，就用哪一行的开场）"


def action_table(heading: str | None = None) -> str:
    """渲染「状态→动作」速查表（5 态、无兜底行、无 [中立]）。

    heading 缺省 = ACTION_TABLE_HEADING（裁判口径，一级标题）。被测客服 prompt 传自己
    的 `## 六、…` 二级标题，让速查表留在「# 本次通话的第一要务」块内、不新增一级块 ——
    内部方案 的分块字数口径因此与 v7/v8 可比。

    状态中文名取 STATE_LABELS（本模块即单一来源）。原 内部脚本 的 5 个值与
    STATE_LABELS 对应项逐字相同（已实测），故搬家后 action_table() 输出与搬家前逐字节一致。
    """
    lines = [
        f"- 听出【{STATE_LABELS[s]}】：{ACTION_V9[s]['opening']}；{ACTION_V9[s]['follow']}。"
        for s in ACTION_TABLE_ORDER
    ]
    head = ACTION_TABLE_HEADING if heading is None else heading
    return head + "\n" + "\n".join(lines)


# ---- 裁判侧渲染（内部方案：被测侧与裁判侧同源，但**判定粒度**必须写清楚）----
# 为什么不直接复用 action_table()：按上面 ACTION_V9 的定义，opening 才是「裁判的唯一判据」，
# follow 只是写进被测 prompt 让回复完整、**不参与判定**。被测侧把 opening+follow 并排渲染是对的
# （模型要照它说话）；裁判侧若照抄同一行，等于把 follow 也升格成必须动作 —— 尺子比被测侧更严，
# 那就是「两侧不同源」换了一副面孔（内部方案 问题 3 要消灭的正是这个）。
# 故裁判侧单独渲染：opening 标为判据、follow 显式标注不单独扣分。数据仍只有 ACTION_V9 一份。
JUDGE_ACTION_HEADING = "各状态对应的动作口径（与被测客服 prompt §六 同源；每行的 opening 才是判据）："


def judge_action_table(heading: str | None = None) -> str:
    """裁判视角的「状态→动作」表：opening = 判据，follow = 参考（不单独扣分）。"""
    lines = [
        f"- 听出【{STATE_LABELS[s]}】：判据＝{ACTION_V9[s]['opening']}"
        f"（后续跟进「{ACTION_V9[s]['follow']}」仅作参考，未做不单独扣分）"
        for s in ACTION_TABLE_ORDER
    ]
    head = JUDGE_ACTION_HEADING if heading is None else heading
    return head + "\n" + "\n".join(lines)


def judge_rubric_text(state: str) -> str:
    """单态一句话 rubric（副指标 action_met 用；与 judge_action_table 同源同粒度）。

    取代 内部脚本 里 v8 措辞的 ACTION_RUBRIC[state]——那份已冻结为
    legacy（仅供 leak()/净化器/历史审阅页复算），裁判路径不得再引用。
    只取 opening，与 内部脚本 的「只核这个开场动作做了没有」同口径。
    """
    code = normalize_state(state) or state
    if code not in ACTION_V9:
        return "按客服规范回应"
    return ACTION_V9[code]["opening"]


# v7 剔除项（键为稳定模板前缀/子串，剔除理由见 AGENT_PROMPT_VERSIONS 注释）
_V7_DROP_KB_PREFIXES = ("用户类型应对要点",)
_V7_DROP_RED_LINE_SUBSTR = "流程顺序推进"


def resolve_agent_prompt_version(variant: str | None = None) -> str:
    """解析生效的 Agent prompt 版本：显式实参 > Settings.agent_prompt（.env AGENT_PROMPT）。"""
    if variant:
        v = str(variant)
    else:
        from listen2serve.runtime.config import get_settings

        v = str(get_settings().agent_prompt)
    if v not in AGENT_PROMPT_VERSIONS:
        raise ValueError(f"未知 Agent prompt 版本 {v!r}，合法值：{AGENT_PROMPT_VERSIONS}")
    return v


def render_customer_profile(
    domain: "DomainSpec | str",
    db_seed: dict[str, Any] | None,
) -> str:
    """将场景 db_seed（单表单行）渲染为自然语言「客户信息档案」（2026-08-14）。

    评测默认不发起 function call（Settings.tools_enabled=False）：原需工具查询的
    用户信息（hotline 订单状态/ETA、marketing 产品名/促销名、collection 欠款事实）
    改由本函数渲染后直接注入初始 prompt；被测侧与裁判契约（policy_judge/
    dialogue_metrics 的 _render_rules）复用本函数，保证同源。

    兼容约定：db_seed 为空/None、预期表缺失或为空、未知域 → 返回空串（旧场景不受影响）。
    列名语义与各域 tables 定义一致（collection: customers；hotline: orders；marketing: products）。

    设计决策：本函数只消费 db_seed 系统侧事实；场景中的用户侧私拟事实
    （user_script.user_facts：还款打算/收入情况/预算顾虑等）不注入客服档案——
    客服侧不应预先知道用户的还款计划，该信息仅由用户模拟器演绎 prompt 消费。
    """
    role = domain.role if isinstance(domain, DomainSpec) else str(domain)
    renderer = _PROFILE_RENDERERS.get(role)
    if renderer is None or not db_seed:
        return ""
    return renderer(db_seed)


# 档案标题：引导语明示「系统已预先提供、通话中无需查询、直接作为事实依据」
_PROFILE_TITLE = "# 客户信息档案（系统已预先提供，通话中无需查询、直接作为事实依据）"


def _fmt_num(value: Any) -> str:
    """数值自然化：整数去小数尾巴（3600.0 → 3600），非数值原样转字符串。"""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _seed_rows(db_seed: dict[str, Any], table: str) -> list[Any]:
    """取预期表的首行（行格式兼容 list/tuple）；缺失或空返回空列表。"""
    rows = db_seed.get(table) or []
    return list(rows[0]) if rows else []


def _profile_collection(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "customers")
    if len(row) < 6:
        return ""
    name, amount, overdue_days, delay_count, promised_date, status = row[:6]
    lines = [
        _PROFILE_TITLE,
        f"客户姓名：{name}",
        f"欠款金额：{_fmt_num(amount)} 元",
        f"逾期天数：{_fmt_num(overdue_days)} 天",
        f"已申请延期次数：{_fmt_num(delay_count)} 次",
        f"上次承诺还款日期：{promised_date}",
        f"当前账户状态：{status}",
        "以上信息通话中可直接使用，向客户提及欠款事实时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


def _profile_hotline(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "orders")
    if len(row) < 5:
        return ""
    order_id, product_name, status, eta, issue_note = row[:5]
    note = issue_note or "无备注"
    lines = [
        _PROFILE_TITLE,
        f"订单号：{order_id}",
        f"商品名称：{product_name}",
        f"订单状态：{status}",
        f"预计送达/完成时间（ETA）：{eta}",
        f"订单备注：{note}",
        "以上信息通话中可直接使用，回答订单相关问题时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


def _profile_marketing(db_seed: dict[str, Any]) -> str:
    row = _seed_rows(db_seed, "products")
    if len(row) < 5:
        return ""
    product_name, category, price, promotion, description = row[:5]
    lines = [
        _PROFILE_TITLE,
        f"产品名称：{product_name}",
        f"产品类别：{category}",
        f"价格：{_fmt_num(price)} 元",
        f"当前活动：{promotion}",
        f"产品简介：{description}",
        "以上信息通话中可直接使用，介绍产品/价格/活动时以本档案为准，不要表述为“正在查询”。",
    ]
    return "\n".join(lines)


_PROFILE_RENDERERS: dict[str, Any] = {
    "collection": _profile_collection,
    "hotline": _profile_hotline,
    "marketing": _profile_marketing,
}


def render_policy_sections(ap: dict[str, Any], variant: str | None = None) -> list[str]:
    """渲染 B6 新增板块（仅渲染 ap 中实际存在的键）。

    被测 prompt 与裁判契约共用本函数，保证同源；旧格式场景（无新键）返回空列表。
    variant="v6" 输出与 内部方案 基线逐字节一致；"v7"/"v9"/"v9_notable" 走精简口径
    （见 AGENT_PROMPT_VERSIONS 与 _LITE_POLICY_VERSIONS）。

    ⚠️ "v8" **不在**白名单里，会继续掉进下面的 v6 全量分支 —— 这就是 内部方案 登记的
    variant 透传缺陷。故意冻结不修：一改 v8 的字节就变，历史 run 的 prompt hash 全部对不上
    （内部方案 V7）。新实验一律用 v9（= 修好缺陷后的「v7 + 速查表」）。
    """
    if resolve_agent_prompt_version(variant) in _LITE_POLICY_VERSIONS:
        return _render_policy_sections_v7(ap)
    blocks: list[str] = []
    if ap.get("success_criteria"):
        blocks.append("## 通话目标与收口标准")
        blocks.extend(f"- {c}" for c in ap["success_criteria"])
    if ap.get("business_flow"):
        # 流程渲染头：目标导向表述（标准流程可跳转与回补，不做硬性阶段闸）
        blocks.append("## 业务流程（以下为标准流程，按客户反应灵活推进，可跳转与回补，每句话服务当前目标）")
        for st in ap["business_flow"]:
            blocks.append(f"### {st.get('id', '')} {st.get('name', '')}")
            blocks.append(f"- 目标：{st.get('goal', '')}")
            blocks.append(f"- 前置：{st.get('precondition', '')}")
            blocks.extend(f"- 要点：{k}" for k in st.get("key_points") or [])
            blocks.append(f"- 出口标准：{st.get('exit_criteria', '')}")
            blocks.append(f"- 未达成时：{st.get('on_fail', '')}")
    if ap.get("red_lines"):
        blocks.append("## 红线（一票否决，任何情况下不得触碰）")
        blocks.extend(f"- {r}" for r in ap["red_lines"])
    if ap.get("knowledge_base"):
        blocks.append("## 业务知识（回答客户咨询必须依据以下口径，禁止编造）")
        blocks.extend(f"- {k}" for k in ap["knowledge_base"])
    if ap.get("speech_style"):
        blocks.append("## 说话风格")
        blocks.extend(f"- {s}" for s in ap["speech_style"])
    return blocks


def _render_policy_sections_v7(ap: dict[str, Any]) -> list[str]:
    """v7 精简板块：把继承自文本客服策略的篇幅让给「听→做」任务面。

    三处剔除都以模板稳定串为键，且只在渲染期生效（场景数据不动，v6 随时可复算）：
    业务流程折为一行一阶段（前置/要点/出口/未达成时不再进 prompt，避免逐阶段硬闸）；
    含「流程顺序推进」的红线剔除（与 AGENT_STATE_FIRST_RULE 直接矛盾）；
    「用户类型应对要点」知识条剔除（是按文字线索索引的旧状态表，与声音状态表竞争）。
    """
    blocks: list[str] = []
    if ap.get("success_criteria"):
        blocks.append("## 本通电话的收口标准")
        blocks.extend(f"- {c}" for c in ap["success_criteria"])
    if ap.get("business_flow"):
        blocks.append("## 业务流程（推进顺序参考，可跳转与回补；与客户当前状态冲突时以状态应对为先）")
        for st in ap["business_flow"]:
            blocks.append(f"- {st.get('id', '')} {st.get('name', '')}：{st.get('goal', '')}")
    red_lines = [r for r in ap.get("red_lines") or [] if _V7_DROP_RED_LINE_SUBSTR not in r]
    if red_lines:
        blocks.append("## 红线（一票否决，任何情况下不得触碰）")
        blocks.extend(f"- {r}" for r in red_lines)
    kb = [k for k in ap.get("knowledge_base") or [] if not k.startswith(_V7_DROP_KB_PREFIXES)]
    if kb:
        blocks.append("## 业务知识（回答客户咨询必须依据以下口径，禁止编造）")
        blocks.extend(f"- {k}" for k in kb)
    return blocks


def render_judge_contract(
    domain: "DomainSpec",
    scenario: dict[str, Any],
    current_state: str | None = None,
    variant: str | None = None,
) -> str:
    """渲染裁判视角的单一策略契约（B2：裁判与被测模型同源）。

    内容 = agent_policy 全字段（角色/沟通风格/必需/禁止/允许/声音要求）
    + 域 policy_rules（规则键归一化为 canonical 英文码，标注当前状态）
    + general_constraints + 声音要求（场景级优先，域级回退）。
    B6：agent_policy 新可选键（business_flow/red_lines/knowledge_base/
    success_criteria/speech_style）经 render_policy_sections 同源纳入；
    内部方案（v6.0）：场景级裁判锚点改为 expected_key_behavior（按关键事件语义描述
    客服应对预期，不绑轮次，仅裁判可见）；旧锚点（stage_at_critical_turn/
    expected_stage_behavior）保留渲染以兼容旧数据重评。

    归一化行为与 B1 一致：旧格式中文键（存量数据/旧 run 重评）经
    normalize_state 归一后命中，不会因格式分裂失配。
    """
    ap = scenario.get("agent_policy") or {}
    version = resolve_agent_prompt_version(variant)
    lines: list[str] = ["# 策略契约（裁判与被测模型同源）"]
    lines.append("## 角色")
    lines.append(ap.get("role_description") or f"{domain.role}客服（{domain.role_type}）")
    if ap.get("communication_style"):
        lines.append("")
        lines.append("## 沟通风格")
        lines.append(str(ap["communication_style"]))
    lines.append("")
    lines.append("## 场景动作约束")
    if ap.get("required_actions"):
        lines.append("必需动作：" + "；".join(ap["required_actions"]))
    if ap.get("forbidden_actions"):
        lines.append("禁止动作：" + "；".join(ap["forbidden_actions"]))
    if ap.get("allowed_actions"):
        lines.append("允许动作：" + "；".join(ap["allowed_actions"]))
    cur = normalize_state(current_state) if current_state else None
    cur_head = (
        f"（当前用户状态：{cur}（{STATE_LABELS[cur]}））" if cur and cur in STATE_LABELS
        else (f"（当前用户状态：{cur}）" if cur else "")
    )
    lines.append("")
    state_head = "## 状态应对规则" if version == "v6" else "## 状态应对规则（客服须以客户声音为主要判据识别状态）"
    lines.append(state_head + cur_head)
    for state, rules in domain.policy_rules.items():
        code = normalize_state(state) or state
        tag = f"{code}（{STATE_LABELS[code]}）" if code in STATE_LABELS else str(code)
        mark = " ←当前状态" if cur and code == cur else ""
        lines.append(f"[{tag}]{mark} 必须：{'/'.join(rules.get('required', []))}")
        if rules.get("forbidden"):
            lines.append(f"[{tag}] 禁止：{'/'.join(rules.get('forbidden', []))}")
        if rules.get("allowed"):
            lines.append(f"[{tag}] 可选：{'/'.join(rules.get('allowed', []))}")
    constraints = list(domain.general_constraints)
    if version in _NO_PLAYBOOK_VERSIONS:
        # v9：裁判契约与被测 prompt 同时瘦身（_v9_filter_constraints 两侧共用同一份写死清单）。
        # 不同步就会反过来造出新错判：裁判拿被测 prompt 里已经不存在的约束去判禁止触发。
        constraints = _v9_filter_constraints(domain.role, constraints, ap.get("red_lines"))
    if version in _LITE_POLICY_VERSIONS:
        # 与被测 prompt 同源（runtime/agent.py 的 v7/v9 指令逐字节复用同三条常量）。
        # 本判断必须跟着白名单走，不能写死 `== "v7"`：否则 Settings.agent_prompt 一旦落到 v9，
        # 裁判契约会静默丢掉这三条规则。v6/v8 不进白名单，输出字节与改动前一致。
        constraints += [AGENT_DUAL_RESPONSE_RULE, AGENT_STATE_FIRST_RULE, AGENT_CLOSING_RULE]
    if constraints:
        lines.append("")
        lines.append("## 通用约束（跨状态）")
        lines.extend(f"- {c}" for c in constraints)
    # B6：同源新板块（仅新格式场景存在；旧格式输出逐字节不变）
    section_lines = render_policy_sections(ap, variant=version)
    if section_lines:
        lines.append("")
        lines.append("## 业务流程与收口（与被测模型同源）")
        lines.extend(section_lines)
    # 内部方案（v6.0）：关键事件应对锚点（不绑轮次，仅裁判可见）
    if scenario.get("expected_key_behavior"):
        lines.append("")
        lines.append("## 关键事件应对锚点（仅裁判可见）")
        lines.append(f"- 关键事件处的预期应对：{scenario['expected_key_behavior']}")
        lines.append("- 说明：收口标准（success_criteria）本轮仅作裁判锚点，不参与自动打分")
    elif scenario.get("stage_at_critical_turn"):
        lines.append("")
        lines.append("## 关键轮流程锚点（仅裁判可见）")
        lines.append(f"- 关键轮所处流程阶段：{scenario['stage_at_critical_turn']}")
        if scenario.get("expected_stage_behavior"):
            lines.append(f"- 该阶段预期行为：{scenario['expected_stage_behavior']}")
        lines.append("- 说明：收口标准（success_criteria）本轮仅作裁判锚点，不参与自动打分")
    lines.append("")
    lines.append("## 声音要求")
    lines.append(ap.get("voice_requirements") or domain.default_voice_requirements or "（无）")
    return "\n".join(lines)


@dataclass
class ToolSpec:
    """可执行服务动作（工具）定义。"""

    name: str
    description: str
    parameters: dict[str, Any] # JSON Schema 参数定义
    read_only: bool = True
    impl: Callable[..., str] | None = field(default=None, repr=False) # (db, **kwargs) -> str


@dataclass
class DBTableSpec:
    """DB 表定义（sqlite 建表语句）。"""

    name: str
    create_sql: str


@dataclass
class DomainSpec:
    """一个客服域的完整定义。"""

    role: str # collection | marketing | hotline
    role_type: str # outbound_pressure | outbound_negotiation | inbound_care
    tools: list[ToolSpec] = field(default_factory=list)
    tables: list[DBTableSpec] = field(default_factory=list)
    # 策略规则：状态 → (必需动作, 禁止动作, 可选动作)
    policy_rules: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # 通用约束（跨状态）
    general_constraints: list[str] = field(default_factory=list)
    default_voice_requirements: str = ""

    # ---- policy 渲染（论文 4.1.7 固定格式）----
    def render_policy_prompt(
        self,
        agent_policy: dict[str, Any] | None = None,
        turn_budget: dict[str, Any] | None = None,
        variant: str | None = None,
    ) -> str:
        """渲染被测 Agent 的 system prompt 策略指令。

        B6：agent_policy 新可选键（business_flow/red_lines/knowledge_base/
        success_criteria/speech_style）按存在性渐进渲染为独立板块；
        无新字段的旧格式场景输出与 B5 基线逐字节一致。
        内部方案（v6.0）：turn_budget.soft 存在时末尾追加节奏引导行（轮次放开后
        引导客服在预算内走完业务流程）；缺省输出不变。
        v7（见 AGENT_PROMPT_VERSIONS）：状态判据显式声明为声音，板块走精简口径；
        「听→做」任务面与收尾轮规则由 runtime/agent.py 的 v7 通话指令承载。
        v9/v9_notable：板块口径同 v7，另加三处瘦身 —— 不再渲染
        「# 用户状态应对策略」整节（状态→动作口径统一到 agent.py 的同源速查表）、
        剔除裁判不判的字数约束、与红线重复的通用约束只留红线那份。
        """
        ap = agent_policy or {}
        version = resolve_agent_prompt_version(variant)
        lines: list[str] = ["# 角色"]
        lines.append(ap.get("role_description", f"你是{self.role}客服，{self.role_type}。"))
        lines.append("")
        lines.append("# 沟通风格")
        lines.append(ap.get("communication_style", ""))
        # B6：通话目标/业务流程/红线/业务知识/说话风格（仅存在时渲染，插入既有骨架）
        section_lines = render_policy_sections(ap, variant=version)
        if section_lines:
            lines.append("")
            lines.append("# 通话目标与业务规范")
            lines.extend(section_lines)
        lines.append("")
        # v9：本节整节不再渲染—— 状态→动作口径统一到 runtime/agent.py 的
        # 「## 六、状态→动作速查表」，那份表与裁判侧同源 import 本模块的 action_table()。
        # 三份规范并存（v7 指令的抽象原则 / v8 速查表 / 本节场景三段式）是 κ 上不去的结构性
        # 原因，也是 内部方案 举的 [配合] 一格互相打架的来源。
        # 只停渲染：self.policy_rules 与场景数据一律不动，v6/v7/v8 随时可复算。
        # 上面那行 lines.append("") 保留 —— 它现在是「# 通话目标与业务规范」与「# 声音要求」
        # 之间唯一的空行分隔，删了会改块结构。
        if version not in _NO_PLAYBOOK_VERSIONS:
            if version == "v6":
                lines.append("# 用户状态应对策略")
            else:
                lines.append("# 用户状态应对策略（状态以你听到的声音为主要判据；本节优先于上方业务流程）")
            for state, rules in self.policy_rules.items():
                # 中文标题由 STATE_LABELS 派生（单一来源；规则键为 canonical 英文码）
                label = STATE_LABELS.get(normalize_state(state) or "", state)
                if version == "v6":
                    lines.append(f"## 当用户表现出[{label}]时：")
                else:
                    lines.append(f"## 当你从客户声音中听出[{label}]时：")
                lines.append("### 必须执行：")
                lines.extend(f"- {a}" for a in rules.get("required", []))
                lines.append("### 禁止行为：")
                lines.extend(f"- {a}" for a in rules.get("forbidden", []))
                lines.append("### 可选动作：")
                lines.extend(f"- {a}" for a in rules.get("allowed", []))
                lines.append("")
        # 声音要求与状态无关，单次渲染（此前逐状态循环内重复渲染；
        # 裁判契约 render_judge_contract 侧本就是单次渲染，两者同源）；
        # v5.15 整改（Ryan 低）：标题提级为与「# 通用约束」平级（旧「### 声音要求：」）
        lines.append("# 声音要求")
        lines.append(ap.get("voice_requirements", self.default_voice_requirements))
        lines.append("")
        lines.append("# 通用约束")
        constraints = self.general_constraints
        if version in _NO_PLAYBOOK_VERSIONS:
            # 内部方案：剔裁判不判的字数约束 + 与本场景红线去重。
            # 与 render_judge_contract 共用 _v9_filter_constraints，两侧同步瘦身。
            constraints = _v9_filter_constraints(self.role, constraints, ap.get("red_lines"))
        for c in constraints:
            lines.append(f"- {c}")
        # 内部方案（v6.0）：节奏引导（用户侧轮次不预定，软预算双侧可见）
        soft = (turn_budget or {}).get("soft")
        if soft:
            lines.append("")
            if version == "v6":
                lines.append(
                    f"本通电话预计约 {soft} 轮内完成主要流程，请合理推进，既不拖沓也不跳步。"
                )
            else:
                lines.append(
                    f"本通电话预计约 {soft} 轮内完成主要流程，请合理推进不要拖沓，"
                    "并为收尾轮留出余量。"
                )
        return "\n".join(lines)

    # ---- DB ----
    def init_db(self, seed_data: dict[str, list[tuple]] | None = None) -> sqlite3.Connection:
        """创建内存 sqlite DB 并注入种子数据。"""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        for t in self.tables:
            conn.execute(t.create_sql)
        for table, rows in (seed_data or {}).items():
            if not rows:
                continue
            placeholders = ",".join("?" * len(rows[0]))
            conn.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        conn.commit()
        return conn

    def describe_db(self) -> str:
        """供 LLM 看到的 DB 结构说明。"""
        return "\n".join(
            f"- {t.name}: {t.create_sql.split('(', 1)[1].rsplit(')', 1)[0]}" for t in self.tables
        )

    # ---- 工具执行 ----
    def execute_tool(self, db: sqlite3.Connection, name: str, args: dict[str, Any]) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            return f"ERROR: 未知工具 {name}"
        if tool.impl is None:
            return f"ERROR: 工具 {name} 未实现"
        try:
            return tool.impl(db, **args)
        except Exception as exc: # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}"

    # ---- 任务装载 ----
    @staticmethod
    def load_tasks(scenarios_path: str | Path) -> list[dict[str, Any]]:
        """读取 data/benchmark/scenarios.jsonl。"""
        path = Path(scenarios_path)
        if not path.exists():
            return []
        tasks = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks
