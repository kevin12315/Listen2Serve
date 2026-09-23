"""TTS 文本规范化（内部方案，v6.1）：数字/金额/单位读音改写 + 多音字高危词替换。

唯一入口 normalize_tts_text(text)：在用户模拟器合成前改写送 TTS 的文本，
不改动落盘对话文本（user_text/评测链路不受影响，词表校验仍对原文）。

改写两类：
1. 数字读音：金额/百分比/日期/序数/量词/编号码 → 中文读法，避免 TTS 逐位
   机械读（如 3600 元读成「三六零零元」）；
2. 多音字高危词（≤50 条起步）：替换为读音无歧义的近义表达（语义基本不变），
   规避常见误读（如「还款」读 hái、「处理」读 chù）。

确定性纯函数，可重入；未知形态原样保留（宁不改，不改错）。
"""

from __future__ import annotations

import re

_DIGITS = "零一二三四五六七八九"


def _int_cn(n: int) -> str:
    """整数中文读法（0-99999999，覆盖金额/日期/编号量级）。"""
    if n == 0:
        return "零"
    parts: list[str] = []
    yi, rem = divmod(n, 100000000)
    wan, rem = divmod(rem, 10000)
    if yi:
        parts.append(f"{_int_cn(yi)}亿")
    if wan:
        if yi and wan < 1000:
            parts.append("零")
        parts.append(f"{_int_cn(wan)}万")
    if rem or not parts:
        if parts and rem < 1000:
            parts.append("零")
        parts.append(_below_wan(rem))
    return "".join(parts)


def _below_wan(n: int) -> str:
    """0-9999 读法（千/百/十位，内部零省略遵循口语习惯）。"""
    if n == 0:
        return ""
    s = ""
    qian, rem = divmod(n, 1000)
    bai, rem2 = divmod(rem, 100)
    shi, ge = divmod(rem2, 10)
    if qian:
        s += _DIGITS[qian] + "千"
    if bai:
        s += _DIGITS[bai] + "百"
    elif qian and (shi or ge):
        s += "零"
    if shi:
        if not (qian or bai) and shi == 1:
            s += "十" # 一十 → 十（口语）
        else:
            s += _DIGITS[shi] + "十"
    elif bai and ge:
        s += "零"
    if ge:
        s += _DIGITS[ge]
    return s


def _money(m: re.Match) -> str:
    """金额改写：3600 元 → 三千六百元；99.9 元 → 九十九点九元。"""
    num = m.group(1)
    if "." in num:
        zheng, xiao = num.split(".", 1)
        xiao_read = "".join(_DIGITS[int(d)] for d in xiao if d.isdigit())
        return f"{_int_cn(int(zheng))}点{xiao_read}元"
    return f"{_int_cn(int(num))}元"


def _percent(m: re.Match) -> str:
    num = m.group(1)
    if "." in num:
        zheng, xiao = num.split(".", 1)
        return f"百分之{_int_cn(int(zheng))}点{''.join(_DIGITS[int(d)] for d in xiao)}"
    return f"百分之{_int_cn(int(num))}"


def _date_dash(m: re.Match) -> str:
    """2025-05-28 → 二零二五年五月二十八日。"""
    y, mo, d = m.group(1), m.group(2), m.group(3)
    y_read = "".join(_DIGITS[int(c)] for c in y)
    return f"{y_read}年{_int_cn(int(mo))}月{_int_cn(int(d))}日"


def _di_ordinal(m: re.Match) -> str:
    """第10 → 第十。"""
    return f"第{_int_cn(int(m.group(1)))}"


_UNIT_CHARS = "天日号年月周次轮个位笔件套条场通"


def _num_unit(m: re.Match) -> str:
    """数字+量词：10 天 → 十天（量词前不加空格）。"""
    return f"{_int_cn(int(m.group(1)))}{m.group(2)}"


def _code_digits(m: re.Match) -> str:
    """编号码中的数字逐位读（ORD0001 → ORD零零零一；工号12345 → 工号一二三四五）。"""
    prefix, digits = m.group(1), m.group(2)
    return prefix + "".join(_DIGITS[int(c)] for c in digits)


# ---- 多音字高危词（≤50 条起步）：误读风险 → 读音无歧义的近义替换 ----
# 只收客服/用户对话高频且公认易误读、且替换不破坏句法搭配者；
# 搭配敏感词（如「感觉/勉强/提供」）一律进豁免表不替换，宁不改不改错。
POLYPHONE_REPLACE: dict[str, str] = {
    "还款": "归还", # huán → 常误读 hái（催收域高频）
    "偿还": "归还", # cháng
    "还清": "归还清", # huán
    "处理": "办理", # chǔ → 常误读 chù（hotline 高频）
    "处分": "处罚", # chǔ
    "卡住": "夹住", # qiǎ → 常误读 kǎ
    "角色": "人物", # jué → 常误读 jiǎo
    "倔强": "顽固", # jué jiàng
    "背包": "背的包", # bēi
    "弯曲": "弯的", # qū
    "曲解": "歪曲理解", # qū
    "投降": "认输", # xiáng
    "咀嚼": "嚼", # jué
    "量体温": "测体温", # liáng
    "丈量": "测量", # liáng
    "宁愿": "情愿", # nìng
    "宁可": "情愿", # nìng
    "铺位": "床位", # pù
    "缝补": "补缀", # féng
    "出差": "外出办事", # chāi
    "剥削": "压榨", # bō xuē
    "薄弱": "单薄", # bó
    "停泊": "停靠", # bó
    "湖泊": "湖", # pō
    "伺候": "照料", # cì
    "便宜": "实惠", # pián yi → 常误读 biàn
    "参差不齐": "长短不齐", # cēn cī
    "差错": "错误", # chā
    "押解": "押送", # jiè
    "给予": "给", # jǐ → 常误读 gěi
    "供给": "供应", # gōng jǐ
}
# 豁免：口语高频但替换会伤自然度/破坏搭配的词不替换（登记备查）
POLYPHONE_EXEMPT = ("还有", "还是", "还行", "得了", "了解", "重新", "感觉",
                    "勉强", "提供", "倒水", "闷热")


def _polyphone(text: str) -> str:
    for word, repl in POLYPHONE_REPLACE.items():
        if word != repl: # 「仅登记」条目（word==repl）跳过
            text = text.replace(word, repl)
    return text


def normalize_tts_text(text: str) -> str:
    """TTS 合成前文本规范化（确定性纯函数）。"""
    if not text:
        return text
    s = text
    # 日期（先于普通数字规则，避免年份被量词规则截获）
    s = re.sub(r"(\d{4})-(\d{1,2})-(\d{1,2})", _date_dash, s)
    # 金额与百分比
    s = re.sub(r"(\d+(?:\.\d+)?)\s*元", _money, s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*[%％]", _percent, s)
    # 序数
    s = re.sub(r"第(\d+)", _di_ordinal, s)
    # 编号码（拉丁/汉字前缀 + 数字串：ORD0001 / 工号12345）
    s = re.sub(r"([A-Za-z]{2,}|工号|订单号|编号)(\d{3,})", _code_digits, s)
    # 数字 + 量词
    s = re.sub(rf"(\d+)\s*([{_UNIT_CHARS}])", _num_unit, s)
    # 多音字高危词
    s = _polyphone(s)
    return s
