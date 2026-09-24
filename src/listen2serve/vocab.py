"""透明度层级词表与校验（数据构造 / 契约测试 / 模拟器模型比较三方共用）。

单一来源原则：（未发布） 生成、tests 契约复验、evaluation.sim_compare /
lexicon_check 词表合规指标均 import 本模块，避免词表多处漂移。

层级语义（论文 4.1.3 / 方案 5.3）：
- T1 explicit：必须命中 ≥1 个明确状态词（按状态极性查对应词表）
- T2 implicit_consistent：仅允许轻微信号词，禁止命中任何情绪词（正/负均禁）
- T3 prosody_only：纯业务中性词汇，禁止命中任何情绪词

版本沿革：
- v4：NEG 50 / POS 30，validate_level 单一校验；
- B4（v5-tpl）：NEG 50→107 / POS 30→65，并按情绪族建状态子表
  （NEG_BY_STATE / POS_BY_STATE；平铺表 NEG_WORDS / POS_WORDS 为子表展平去重，
  v4 全部旧词保留、语义不变）；T2_SIGNAL_WORDS 扩充并新增 T2_HEDGE_PATTERNS /
  T3_BUSINESS_WORDS。新增 validate_level_v2 与旧函数**双校验并行**（旧函数
  签名与语义完全不动，sim_compare / lexicon_check / datagen 三方共用不受影响）：
  * T1：须命中对应**状态子表** ≥1 词（较旧版按极性查平铺表更严）；
  * T2：不命中情绪词 + 白名单下限（≥1 信号词或缓和句式）；
  * T3：不命中情绪词 + 业务中性白名单下限。
- B4 已知词表边界（语境豁免仅对 v2 生效）：道别语「先挂了」会被平铺表「挂了」
  误报违规（B3 冒烟 T2 补跑实测发现）；v2 匹配器先剔除豁免短语再查词，
  旧 validate_level 语义保持不变，该边界在其口径下保留并在审计记录标注。
"""

from __future__ import annotations

# ---- 状态极性（内部方案 v6.0：新 6 态码不再带 N/P 前缀，极性由本表声明）----
# neutral 无极性子表（T1 关键轮若为 neutral 态不设词表要求，与旧 N0 口径一致）。
STATE_POLARITY: dict[str, str] = {
    "displeased": "NEG",
    "urgent": "NEG",
    "doubtful": "NEG",
    "hushed": "NEG",
    "cooperative": "POS",
}

# ---- 负面明确状态词（内部方案：子表键按新 6 态合并重排，词本体零改动）----
NEG_BY_STATE: dict[str, list[str]] = {
    # 不满族（旧 N1 烦躁恼怒 + N5 抗拒无力合并；情绪面同策略：安抚+给台阶）
    "displeased": [
        "烦", "恼", "怒", "骂", "讨厌", "无语", "受不了", "崩溃", "委屈", "失望",
        "太累", "生气", "气人", "火大", "闹心", "添乱", "烦死", "不耐烦", "折腾", "嫌",
        "受够", "窝火", "憋气", "来气", "气炸", "上火", "烦躁", "心烦", "气不过",
        "投诉", "没钱", "挂了", "算了", "认了", "没用", "白干", "滚", "敷衍", "拖着",
        "还不起", "拿不出", "不接茬", "没兴趣", "不需要", "用不着", "别再来",
        "免谈", "没门", "不想掏", "休想", "懒得理",
    ],
    # 急迫族（旧 N2 急切催促）
    "urgent": [
        "急", "着急", "赶时间", "来不及", "催", "等不及", "急死", "火烧眉毛",
        "十万火急", "急得慌", "耽搁不起", "火急火燎", "没工夫",
    ],
    # 疑问/不信族（旧 N3 困惑不解 + N6 怀疑警惕合并；同为「解释清楚、给凭证」）
    "doubtful": [
        "搞不懂", "听不懂", "什么玩意", "糊涂", "绕晕", "一头雾水", "摸不着头脑",
        "云里雾里", "听不明白", "理不清", "整不明白", "晕头转向", "稀里糊涂",
        "骗子", "坑", "凭什么", "害怕", "吓", "忽悠", "上当", "信不过", "糊弄",
        "不敢信", "套路", "猫腻", "来路不明", "半信半疑", "不放心", "有诈", "冒牌",
    ],
    # 轻声/不便族（旧 N4 隐私不便/被打扰）
    "hushed": [
        "别打", "没时间", "不方便", "骚扰", "没完没了", "逼我", "没空", "顾不上",
        "正忙着", "不是时候", "打扰", "别这时候", "抽不出空",
    ],
}

# ---- 正面明确状态词（内部方案：四个 P 族合并为 cooperative，词本体零改动）----
POS_BY_STATE: dict[str, list[str]] = {
    # 配合族（旧 P1 配合顺从 + P2 放松从容 + P3 感激满意 + P4 兴趣心动）
    "cooperative": [
        "听懂了", "明白了", "愿意配合", "没问题", "好说", "爽快", "都配合",
        "你说咋办就咋办", "按你说的来", "都听你的", "需要啥就说", "尽管开口",
        "全力支持", "没二话", "依你", "配合你们",
        "不着急", "不忙", "有空", "慢慢说", "慢慢讲", "有的是时间", "不急这一会儿",
        "时间还早", "现在方便", "闲着呢", "不着忙",
        "满意", "放心了", "谢谢你们", "感谢", "靠谱", "帮大忙", "省心", "周到",
        "热心", "耐心", "舒心", "安心", "贴心", "给力", "感激", "认可", "信得过",
        "放心", "多亏你们", "办得漂亮", "费心了", "辛苦你们", "记你们的好",
        "真不错", "挺满意", "放心交给你们",
        "感兴趣", "太好了", "乐意", "挺乐意", "听着不错", "想了解", "挺心动",
        "划算", "值得办", "有点意思", "越说越想办", "有吸引力",
    ],
}


def _flatten(tables: dict[str, list[str]]) -> list[str]:
    """状态子表展平去重（保序）→ 平铺表（单一真相源：平铺表恒等于子表并集）。"""
    seen: set[str] = set()
    flat: list[str] = []
    for words in tables.values():
        for w in words:
            if w not in seen:
                seen.add(w)
                flat.append(w)
    return flat


NEG_WORDS: list[str] = _flatten(NEG_BY_STATE) # 内部方案：107 词（成员不变，归属重排）
POS_WORDS: list[str] = _flatten(POS_BY_STATE) # 内部方案：65 词（成员不变，归属重排）


def state_pool(state: str) -> list[str] | None:
    """状态对应的 T1 词表子表（内部方案 新码）；neutral/未知态返回 None（无词表要求）。"""
    if state in NEG_BY_STATE:
        return NEG_BY_STATE[state]
    if state in POS_BY_STATE:
        return POS_BY_STATE[state]
    return None


def polarity_pool(state: str) -> list[str]:
    """按状态极性返回平铺表（子表缺失时的回落）；neutral/未知态回落 NEG 平铺表。"""
    return POS_WORDS if STATE_POLARITY.get(state) == "POS" else NEG_WORDS

# T2 轻微信号词（允许出现，不计违规；B4 起参与 v2 白名单下限校验）
T2_SIGNAL_WORDS: list[str] = [
    "那个", "就是", "能不能", "嗯", "之前", "差不多", "要不",
    # B4 扩充（口语犹豫 / 铺垫标记）
    "这个", "那什么", "唉", "哎", "你看", "这么说吧",
]

# T2 缓和句式（B4 新增）：与信号词共同构成 T2 白名单下限（命中任一即可）
T2_HEDGE_PATTERNS: list[str] = [
    "能不能", "可不可以", "是不是", "要不", "还是", "再", "先", "稍微", "尽量",
    "一点", "一下", "行不行", "好不好", "方便吗", "行吗",
]

# T3 业务中性白名单（B4 新增）：T3 关键轮须命中 ≥1，证明确为业务表述而非空转
T3_BUSINESS_WORDS: list[str] = [
    "流程", "办理", "确认", "核实", "材料", "条件", "条款", "价格", "方式", "手续",
    "订单", "还款", "产品", "业务", "工号", "渠道", "期限", "账单", "短信", "记录",
    "时间", "金额", "方案", "申请", "审核", "到账", "优惠", "活动", "预约", "合同",
]

# 语境豁免（B4，仅 validate_level_v2 生效）：词 → 命中后不计违规的短语。
# 「先挂了」为道别语，不应被「挂了」误报（B3 冒烟 T2 补跑实测边界）。
NEG_CONTEXT_EXEMPT: dict[str, tuple[str, ...]] = {
    "挂了": ("先挂了",),
}

# ---- 运行期语境豁免表（仅 validate_level_runtime 生效）----
# 口径：非关键轮 T2/T3 演绎话术的事后违规计数（lexicon_check）。
# 条目 =（命中词 + 左右邻接字符模式 [+ 守卫词]），由误报聚类结果整理而成；每条规则先由
# LLM 逐条判定再人工复核（判定所用型号未公开，见 docs/limitations.md），逐条可核查，
# 不引入分词依赖。不影响 validate_level（v4 契约）与
# validate_level_v2（关键轮预写文本契约），两者语义保持不变。
# 规则字段：left/right 为邻接字符串候选（空元组 = 该侧无要求；
# right 中 "" 表示句尾），guard 为句内守卫词（命中任一则豁免失效）。
RUNTIME_CONTEXT_EXEMPT: dict[str, tuple[dict, ...]] = {
    # 道别语：Step 2 误报 52/64（左邻 先/我/，/。 × 右邻 。/啊/句尾全覆盖）；
    # 句内含抗拒/威胁词（投诉/骚扰/别打/再打…）时「挂了」携不满语气，守卫失效；
    # 内部方案 审阅补：左邻增 这边/那我/就（「我这边挂了啊」在 K18 词表误报中出现，Q4）
    "挂了": ({"left": ("先", "我", "这边", "那我", "就", "，", "。"), "right": ("。", "啊", ""),
              "guard": ("投诉", "骚扰", "别打", "再打", "不要再打", "别再来"),
              "source": "内部方案 Step 2 误报聚类：道别语（内部方案 补左邻候选）"},),
    # 否定语境的态度陈述：「我不认可啊」——「认可」在 cooperative 正向子表，否定后
    # 极性反转，计作「命中明确情绪词」属误报（K18 词表审阅 Q4）；态度陈述本身不是
    # 情绪词，T2/T3 的态度约束由 prompt 侧负责，不走词表防线。
    "认可": ({"left": ("不", "不太", "难", "很难", "没法"), "right": (), "guard": (),
              "source": "内部方案 K18 词表审阅：否定语境极性反转"},),
    # 催收业务确认问句：「你们是催这期还款的吧」（Step 2 误报 4/9）
    "催": ({"left": (), "right": ("这期",), "guard": (),
            "source": "内部方案 Step 2 误报聚类：催收业务确认问句"},),
    # 承诺不拖延：「不拖着」（Step 2 误报 3/5；「一直拖着」等抱怨语境不豁免）
    "拖着": ({"left": ("不",), "right": (), "guard": (),
              "source": "内部方案 Step 2 误报聚类：承诺不拖延"},),
    # 客观时间不便陈述：「这会儿/今天 + 不方便」（Step 2 误报 2/2）
    "不方便": ({"left": ("这会儿", "今天"), "right": (), "guard": (),
               "source": "内部方案 Step 2 误报聚类：客观时间不便陈述"},),
}

# 运行期子串型豁免：命中词出现在更长的中性短语内 → 该次出现不计违规
# （B4 既有「先挂了」短语形式保留；新增条目均注明聚类来源）。
RUNTIME_PHRASE_EXEMPT: dict[str, tuple[str, ...]] = {
    "挂了": ("先挂了",), # B4 已知边界（NEG_CONTEXT_EXEMPT 同条）
    "烦": ("麻烦",), # 礼貌用语「麻烦你/麻烦了」（Step 2 误报 5/5）
    "不需要": ("需不需要",), # 中性征疑问句「需不需要我再补材料」（Step 2 误报 1/1）
    # 业务角色名词：「催收方/催收部门/催收安排」里的「催」不是用户的情绪表达
    # （K18d 数据侧 surface 生成实测：3/144 条 T3 中性句仅因含「催收」被判违规；
    # 该误报同样存在于运行期防线 _vocab_defense，会把用户说「你们催收部门」判成
    # T2/T3 情绪词命中并触发无谓重试）。既有 RUNTIME_CONTEXT_EXEMPT["催"] 只豁免
    # 「催这期」这一种邻接，覆盖不到构词。
    # 「催缴」同理（n2 第三轮实测：「本次催缴通知的下发时间」「本次催缴的具体依据」
    # 这类纯业务问句被判违规，2 条连试 3 次都过不了）——「催缴通知」是催收业务
    # 的标准名词，情绪由「别再…」「一直…」这类修饰承载，而那些由 guard 守卫拦。
    "催": ("催收", "催缴"),
}


def t2_whitelist_hit(text: str) -> list[str]:
    """T2 白名单命中清单：信号词 ∪ 缓和句式（B4：T2 隐含线索的下限度量）。"""
    return [w for w in T2_SIGNAL_WORDS + T2_HEDGE_PATTERNS if w in text]


def _strip_exempt(text: str) -> str:
    """按 NEG_CONTEXT_EXEMPT 剔除豁免短语（如道别语「先挂了」），再做词表匹配。"""
    for phrases in NEG_CONTEXT_EXEMPT.values():
        for phrase in phrases:
            text = text.replace(phrase, "")
    return text


def _emotion_hits(text: str, words: list[str]) -> list[str]:
    """v2 口径情绪词命中（带语境豁免）。"""
    scrubbed = _strip_exempt(text)
    return [w for w in words if w in scrubbed]


def _occurrence_runtime_exempt(text: str, word: str, pos: int) -> bool:
    """pos 处 word 的一次命中是否被运行期豁免表覆盖。

    守卫优先：该词任一邻接规则带 guard 且句内命中守卫词（如「挂了」句内含
    投诉/别打等抗拒语境）→ 该词全部豁免（含短语型）失效；
    其余两类规则命中任一即豁免：
    1. 短语覆盖：命中落在 RUNTIME_PHRASE_EXEMPT 某中性短语的出现区间内；
    2. 邻接模式：RUNTIME_CONTEXT_EXEMPT 的 left/right 候选均满足（空元组侧无要求；
       right 含 "" 时允许句尾）。
    """
    for rule in RUNTIME_CONTEXT_EXEMPT.get(word, ()):
        if any(g in text for g in rule.get("guard", ())):
            return False # 守卫词命中（抗拒/威胁语境）→ 该词豁免全部失效
    for phrase in RUNTIME_PHRASE_EXEMPT.get(word, ()):
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            if idx <= pos < idx + len(phrase):
                return True
            start = idx + 1
    for rule in RUNTIME_CONTEXT_EXEMPT.get(word, ()):
        left_ok = not rule["left"] or any(
            text[pos - len(l):pos] == l for l in rule["left"] if l)
        right_ok = not rule["right"] or any(
            (r == "" and pos + len(word) == len(text))
            or (r and text[pos + len(word):pos + len(word) + len(r)] == r)
            for r in rule["right"])
        if left_ok and right_ok:
            return True
    return False


def runtime_emotion_hits(text: str) -> list[str]:
    """运行期口径情绪词命中：NEG+POS 平铺表子串匹配，逐次出现按豁免表豁免。

    与 _emotion_hits（v2 口径，仅剔除 NEG_CONTEXT_EXEMPT 短语）分开：本函数承载
    内部方案 扩展豁免表，仅供非关键轮演绎话术的运行期计数/防线使用。
    """
    hits: list[str] = []
    for w in NEG_WORDS + POS_WORDS:
        start = 0
        while True:
            idx = text.find(w, start)
            if idx < 0:
                break
            if not _occurrence_runtime_exempt(text, w, idx):
                hits.append(w)
                break # 同词存在未豁免命中即计违规（与轮级口径一致）
            start = idx + 1
    return hits


def validate_level_runtime(text: str, level: str, state: str) -> tuple[bool, str]:
    """运行期口径校验：T2/T3 情绪词禁用改用运行期豁免表；
    T1 分支与旧口径完全一致（回落 validate_level）。

    与 validate_level（v4 三方共用契约）/validate_level_v2（关键轮预写文本契约）
    并行存在，不改两者语义；仅用于非关键轮演绎话术的违规计数与演绎侧防线。
    """
    if level in ("T2", "T3"):
        bad = runtime_emotion_hits(text)
        if bad:
            return False, f"命中明确情绪词: {bad}"
        return True, "合规"
    return validate_level(text, level, state)


def validate_level(text: str, level: str, state: str) -> tuple[bool, str]:
    """按透明度层级做词表硬校验。返回 (是否合规, 说明)。

    注意：本函数为 v4 口径，签名与语义保持不变（三方共用契约）；
    状态子表 / 白名单下限 / 语境豁免见 validate_level_v2。
    """
    if level == "T1":
        if STATE_POLARITY.get(state) is None:
            return True, "neutral 态无词表要求"
        pool = polarity_pool(state)
        hit = [w for w in pool if w in text]
        return (True, f"命中: {hit}") if hit else (False, "未命中明确状态词")
    # T2/T3：不得命中任何明确情绪词（T2 额外允许轻微信号词，但信号词不在情绪词表中，无需豁免）
    bad = [w for w in NEG_WORDS + POS_WORDS if w in text]
    if bad:
        return False, f"命中明确情绪词: {bad}"
    return True, "合规"


def validate_level_v2(text: str, level: str, state: str) -> tuple[bool, str]:
    """B4 加强校验（内部方案 起子表键为新 6 态码）：

    - T1：必须命中对应**状态子表** ≥1 词（未知状态回落极性平铺表；
      neutral 态无词表要求）；
    - T2：不命中任何情绪词（带语境豁免）且命中 ≥1 信号词 / 缓和句式；
    - T3：不命中任何情绪词（带语境豁免）且命中 ≥1 业务中性白名单词。
    """
    if level == "T1":
        if STATE_POLARITY.get(state) is None:
            return True, "neutral 态无词表要求"
        pool = state_pool(state) or polarity_pool(state)
        hit = [w for w in pool if w in text]
        return (True, f"状态子表命中: {hit}") if hit else (False, f"未命中 {state} 状态子表词")
    bad = _emotion_hits(text, NEG_WORDS + POS_WORDS)
    if bad:
        return False, f"命中明确情绪词: {bad}"
    if level == "T2":
        wl = t2_whitelist_hit(text)
        return (True, f"白名单命中: {wl}") if wl else (False, "未命中信号词/缓和句式（T2 白名单下限）")
    if level == "T3":
        biz = [w for w in T3_BUSINESS_WORDS if w in text]
        return (True, f"业务白名单命中: {biz}") if biz else (False, "未命中业务中性白名单（T3 下限）")
    return True, "合规"

# 逐轮 TTS 情绪控制已迁至 listen2serve.emotion（B5：STATE_EMOTION / tts_control_for
# 兼容层 + EMOTION_LIBRARY 情绪库 + tts_compose 结构化 instruct；本模块不再持有副本）
