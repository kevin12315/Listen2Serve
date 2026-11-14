"透明度层级词表与校验（数据构造 / 契约测试 / 模拟器模型比较三方共用）"

from __future__ import annotations

STATE_POLARITY: dict[str, str] = {
    "displeased": "NEG",
    "urgent": "NEG",
    "doubtful": "NEG",
    "hushed": "NEG",
    "cooperative": "POS",
}

NEG_BY_STATE: dict[str, list[str]] = {
    "displeased": [
        "烦",
        "恼",
        "怒",
        "骂",
        "讨厌",
        "无语",
        "受不了",
        "崩溃",
        "委屈",
        "失望",
        "太累",
        "生气",
        "气人",
        "火大",
        "闹心",
        "添乱",
        "烦死",
        "不耐烦",
        "折腾",
        "嫌",
        "受够",
        "窝火",
        "憋气",
        "来气",
        "气炸",
        "上火",
        "烦躁",
        "心烦",
        "气不过",
        "投诉",
        "没钱",
        "挂了",
        "算了",
        "认了",
        "没用",
        "白干",
        "滚",
        "敷衍",
        "拖着",
        "还不起",
        "拿不出",
        "不接茬",
        "没兴趣",
        "不需要",
        "用不着",
        "别再来",
        "免谈",
        "没门",
        "不想掏",
        "休想",
        "懒得理",
    ],
    "urgent": [
        "急",
        "着急",
        "赶时间",
        "来不及",
        "催",
        "等不及",
        "急死",
        "火烧眉毛",
        "十万火急",
        "急得慌",
        "耽搁不起",
        "火急火燎",
        "没工夫",
    ],
    "doubtful": [
        "搞不懂",
        "听不懂",
        "什么玩意",
        "糊涂",
        "绕晕",
        "一头雾水",
        "摸不着头脑",
        "云里雾里",
        "听不明白",
        "理不清",
        "整不明白",
        "晕头转向",
        "稀里糊涂",
        "骗子",
        "坑",
        "凭什么",
        "害怕",
        "吓",
        "忽悠",
        "上当",
        "信不过",
        "糊弄",
        "不敢信",
        "套路",
        "猫腻",
        "来路不明",
        "半信半疑",
        "不放心",
        "有诈",
        "冒牌",
    ],
    "hushed": [
        "别打",
        "没时间",
        "不方便",
        "骚扰",
        "没完没了",
        "逼我",
        "没空",
        "顾不上",
        "正忙着",
        "不是时候",
        "打扰",
        "别这时候",
        "抽不出空",
    ],
}

POS_BY_STATE: dict[str, list[str]] = {
    "cooperative": [
        "听懂了",
        "明白了",
        "愿意配合",
        "没问题",
        "好说",
        "爽快",
        "都配合",
        "你说咋办就咋办",
        "按你说的来",
        "都听你的",
        "需要啥就说",
        "尽管开口",
        "全力支持",
        "没二话",
        "依你",
        "配合你们",
        "不着急",
        "不忙",
        "有空",
        "慢慢说",
        "慢慢讲",
        "有的是时间",
        "不急这一会儿",
        "时间还早",
        "现在方便",
        "闲着呢",
        "不着忙",
        "满意",
        "放心了",
        "谢谢你们",
        "感谢",
        "靠谱",
        "帮大忙",
        "省心",
        "周到",
        "热心",
        "耐心",
        "舒心",
        "安心",
        "贴心",
        "给力",
        "感激",
        "认可",
        "信得过",
        "放心",
        "多亏你们",
        "办得漂亮",
        "费心了",
        "辛苦你们",
        "记你们的好",
        "真不错",
        "挺满意",
        "放心交给你们",
        "感兴趣",
        "太好了",
        "乐意",
        "挺乐意",
        "听着不错",
        "想了解",
        "挺心动",
        "划算",
        "值得办",
        "有点意思",
        "越说越想办",
        "有吸引力",
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


NEG_WORDS: list[str] = _flatten(NEG_BY_STATE)
POS_WORDS: list[str] = _flatten(POS_BY_STATE)


def state_pool(state: str) -> list[str] | None:
    "状态对应的 T1 词表子表（ 新码）；neutral/未知态返回 None（无词表要求）"
    if state in NEG_BY_STATE:
        return NEG_BY_STATE[state]
    if state in POS_BY_STATE:
        return POS_BY_STATE[state]
    return None


def polarity_pool(state: str) -> list[str]:
    """按状态极性返回平铺表（子表缺失时的回落）；neutral/未知态回落 NEG 平铺表。"""
    return POS_WORDS if STATE_POLARITY.get(state) == "POS" else NEG_WORDS


T2_SIGNAL_WORDS: list[str] = [
    "那个",
    "就是",
    "能不能",
    "嗯",
    "之前",
    "差不多",
    "要不",
    # B4 扩充（口语犹豫 / 铺垫标记）
    "这个",
    "那什么",
    "唉",
    "哎",
    "你看",
    "这么说吧",
]

# T2 缓和句式（B4 新增）：与信号词共同构成 T2 白名单下限（命中任一即可）
T2_HEDGE_PATTERNS: list[str] = [
    "能不能",
    "可不可以",
    "是不是",
    "要不",
    "还是",
    "再",
    "先",
    "稍微",
    "尽量",
    "一点",
    "一下",
    "行不行",
    "好不好",
    "方便吗",
    "行吗",
]

# T3 业务中性白名单（B4 新增）：T3 关键轮须命中 ≥1，证明确为业务表述而非空转
T3_BUSINESS_WORDS: list[str] = [
    "流程",
    "办理",
    "确认",
    "核实",
    "材料",
    "条件",
    "条款",
    "价格",
    "方式",
    "手续",
    "订单",
    "还款",
    "产品",
    "业务",
    "工号",
    "渠道",
    "期限",
    "账单",
    "短信",
    "记录",
    "时间",
    "金额",
    "方案",
    "申请",
    "审核",
    "到账",
    "优惠",
    "活动",
    "预约",
    "合同",
]

NEG_CONTEXT_EXEMPT: dict[str, tuple[str, ...]] = {
    "挂了": ("先挂了",),
}

RUNTIME_CONTEXT_EXEMPT: dict[str, tuple[dict, ...]] = {
    "挂了": (
        {
            "left": ("先", "我", "这边", "那我", "就", "，", "。"),
            "right": ("。", "啊", ""),
            "guard": ("投诉", "骚扰", "别打", "再打", "不要再打", "别再来"),
            "source": "Reviewed contextual exception",
        },
    ),
    # 否定语境的态度陈述：「我不认可啊」——「认可」在 cooperative 正向子表，否定后
    # 极性反转，计作「命中明确情绪词」属误报（K18 词表审阅 Q4）；态度陈述本身不是
    # 情绪词，T2/T3 的态度约束由 prompt 侧负责，不走词表防线。
    "认可": (
        {
            "left": ("不", "不太", "难", "很难", "没法"),
            "right": (),
            "guard": (),
            "source": "Reviewed contextual exception",
        },
    ),
    # 催收业务确认问句：「你们是催这期还款的吧」（Step 2 误报 4/9）
    "催": (
        {
            "left": (),
            "right": ("这期",),
            "guard": (),
            "source": "Reviewed contextual exception",
        },
    ),
    # 承诺不拖延：「不拖着」（Step 2 误报 3/5；「一直拖着」等抱怨语境不豁免）
    "拖着": (
        {
            "left": ("不",),
            "right": (),
            "guard": (),
            "source": "Reviewed contextual exception",
        },
    ),
    # 客观时间不便陈述：「这会儿/今天 + 不方便」（Step 2 误报 2/2）
    "不方便": (
        {
            "left": ("这会儿", "今天"),
            "right": (),
            "guard": (),
            "source": "Reviewed contextual exception",
        },
    ),
}

# 运行期子串型豁免：命中词出现在更长的中性短语内 → 该次出现不计违规
# （B4 既有「先挂了」短语形式保留；新增条目均注明聚类来源）。
RUNTIME_PHRASE_EXEMPT: dict[str, tuple[str, ...]] = {
    "没用": ("没用几次",),  # 字面使用次数，不表示没有作用。
    "急": ("紧急处理", "紧急工单"),  # 业务处理名称，不是用户状态自述。
    "挂了": ("先挂了",),  # B4 已知边界（NEG_CONTEXT_EXEMPT 同条）
    "烦": ("麻烦",),  # 礼貌用语「麻烦你/麻烦了」（Step 2 误报 5/5）
    "不需要": ("需不需要",),  # 中性征疑问句「需不需要我再补材料」（Step 2 误报 1/1）
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
    "口径情绪词命中（带语境豁免）"
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
            return False  # 守卫词命中（抗拒/威胁语境）→ 该词豁免全部失效
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
            text[pos - len(frag) : pos] == frag for frag in rule["left"] if frag
        )
        right_ok = not rule["right"] or any(
            (r == "" and pos + len(word) == len(text))
            or (r and text[pos + len(word) : pos + len(word) + len(r)] == r)
            for r in rule["right"]
        )
        if left_ok and right_ok:
            return True
    return False


def runtime_emotion_hits(text: str) -> list[str]:
    "运行期口径情绪词命中：NEG+POS 平铺表子串匹配，逐次出现按豁免表豁免"
    hits: list[str] = []
    for w in NEG_WORDS + POS_WORDS:
        start = 0
        while True:
            idx = text.find(w, start)
            if idx < 0:
                break
            if not _occurrence_runtime_exempt(text, w, idx):
                hits.append(w)
                break  # 同词存在未豁免命中即计违规（与轮级口径一致）
            start = idx + 1
    return hits


def validate_level_runtime(
    text: str, level: str, state: str, *, is_key_turn: bool = False
) -> tuple[bool, str]:
    """Check neutral turns for explicit emotion words and state declarations."""
    if level in ("T2", "T3"):
        bad = runtime_emotion_hits(text)
        if bad:
            return False, f"命中明确情绪词: {bad}"
        declarations = _explicit_state_declarations(text, is_key_turn=is_key_turn and level == "T3")
        if declarations:
            return False, f"命中明确状态自述: {declarations}"
        return True, "合规"
    return validate_level(text, level, state)


def _explicit_state_declarations(text: str, *, is_key_turn: bool = False) -> list[str]:
    from listen2serve.data.wording_ban import (
        explicit_key_state_declaration_hits,
        explicit_state_declaration_hits,
    )

    hits = explicit_state_declaration_hits(text)
    if is_key_turn:
        hits += explicit_key_state_declaration_hits(text)
    return list(dict.fromkeys(hits))


def neutral_wording_hits(text: str, *, is_key_turn: bool = False) -> list[str]:
    """Use the runtime lexicon and narrow state-declaration check together."""
    return list(
        dict.fromkeys(
            runtime_emotion_hits(text) + _explicit_state_declarations(text, is_key_turn=is_key_turn)
        )
    )


def validate_level(text: str, level: str, state: str) -> tuple[bool, str]:
    "按透明度层级做词表硬校验。返回 (是否合规, 说明)"
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


def validate_wording(text: str, level: str, state: str) -> tuple[bool, str]:
    """Validate wording cues using the shared per-state vocabulary."""
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
        declarations = _explicit_state_declarations(text, is_key_turn=True)
        if declarations:
            return False, f"命中明确状态自述: {declarations}"
        biz = [w for w in T3_BUSINESS_WORDS if w in text]
        return (True, f"业务白名单命中: {biz}") if biz else (False, "未命中业务中性白名单（T3 下限）")
    return True, "合规"


# 逐轮 TTS 情绪控制已迁至 listen2serve.emotion（B5：STATE_EMOTION / tts_control_for
# 兼容层 + EMOTION_LIBRARY 情绪库 + tts_compose 结构化 instruct；本模块不再持有副本）
