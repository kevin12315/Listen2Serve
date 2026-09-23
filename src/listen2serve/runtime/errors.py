"""runtime 层可辨识异常。

放这里而不是放各 provider 内部，是因为 orchestrator 需要**按类型**分辨断因：
靠 `str(exc)` 匹配文案会在 provider 改一句提示语时静默失效，而这类失效的
后果是「本该留用的轨迹被判失败」——不报错、只少数据，最难发现。
"""

from __future__ import annotations


class EndpointSessionClosed(RuntimeError):
    """被测端点的 realtime 会话被服务端单方面关闭（我方未主动断开）。

    与「连接建立失败」严格区分：本异常只用于**曾经连上并正常对话过**之后
    服务端掐断的情形（如某全双工端点的
    `{"type":"session.closed","reason":"backend_error"}`，实测会话活到
    91–146s 即触发，与回复轮数、输出 token 无关，我方无参数可延长）。
    建连阶段的失败仍抛普通 RuntimeError —— 那是「这通根本没开始」，
    不存在「关键轮是否已跑完」这回事。

    继承 RuntimeError 而非 Exception：既有 `except RuntimeError` 的调用方
    行为不变（纯附加，不改任何现有捕获路径）。
    """
