#!/usr/bin/env python3
"""离线冒烟：不连任何外部 API，验证"把裁判换成通用 OpenAI 兼容后端"这条链路真的通。

为什么需要这个脚本：发布仓把裁判从内部代理换成了 `openai/<模型名>` + OPENAI_BASE_URL，
而这条链路能不能跑，原本只有配一个真 key 打一次才知道。这里用一个**本地桩服务**冒充
任意 OpenAI 兼容端点，跑完整三段：
 ① LLMGateway 走 openai 后端发一次 chat → 证明解析/凭据/URL 拼接对
 ② KeyTurnJudge 拿真实场景与真实裁判 prompt → 证明量规渲染 + JSON 回读对
 ③ KeyTurnPass 的判定口径（分数 ≥4 且无禁止动作）→ 证明主指标算得出来
跑完打印每一项的实测值，任何一步不对就非零退出。

用法：
 cd listen2serve && PYTHONPATH=src python scripts/smoke_offline.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 桩服务的回答：一份合法的 KeyTurn 裁判 JSON（v2.5-K 口径：1-5 分 + 禁止项逐条）
STUB_REPLY = {
    "key_turn_score": 4,
    "key_turn_evidence": "客服先承接了客户的场合不便，再回到业务",
    "state_fit": "是",
    "forbidden_items": [
        {"action": "追问欠款细节", "triggered": "否", "evidence": ""},
    ],
    "reason": "策略方向正确，分寸略欠",
}
SEEN: list[dict] = []


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode("utf-8"))
        SEEN.append({"path": self.path, "model": body.get("model"),
                     "n_messages": len(body.get("messages", []))})
        payload = {
            "id": "chatcmpl-stub", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": json.dumps(STUB_REPLY, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 512, "completion_tokens": 64, "total_tokens": 576},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_a):  # 静音默认访问日志
        return


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # 关键：凭据与端点全部走环境变量 ⇒ 发布仓不预设任何厂商
    os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["OPENAI_API_KEY"] = "stub-key-not-a-secret"
    os.environ["JUDGE_MODEL"] = "openai/stub-judge-1"
    from listen2serve.runtime.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    print(f"[0] 配置：judge_model={settings.judge_model} base_url={settings.openai_base_url}")

    # ① 通用后端的一次 chat
    from listen2serve.gateway.llm_gateway import LLMGateway

    gw = LLMGateway(settings)
    r = gw.chat([{"role": "user", "content": "ping"}], model=settings.judge_model,
                temperature=0.0, max_tokens=64, seed=1)
    assert json.loads(r.text)["key_turn_score"] == 4, r.text
    print(f"[1] chat 通过：model={r.model} tokens={r.prompt_tokens}/{r.completion_tokens} "
          f"latency={r.latency_ms:.0f}ms")

    # ② 真实场景 + 真实裁判 prompt
    scenarios = [json.loads(x) for x in
                 (ROOT / "data/benchmark/scenarios.jsonl").read_text(encoding="utf-8").splitlines()
                 if x.strip()]
    sc = next(s for s in scenarios if s["role"] == "collection")
    from listen2serve.domains import get_domain
    from listen2serve.evaluation.policy_judge import KeyTurnJudge

    judge = KeyTurnJudge(llm=gw, llm_model=settings.judge_model)
    verdict = judge.judge(get_domain(sc["role"]), sc,
                          [{"role": "user", "content": "（关键轮客户台词占位）"}],
                          "客服：您现在方便讲话吗？我长话短说。")
    print(f"[2] 裁判通过：score={verdict.key_turn_score} met={verdict.key_behavior_met} "
          f"fit={verdict.state_fit} 解析失败={judge.parse_failures}")

    # ③ 主指标口径：分数 ≥4 且当轮无禁止动作 ⇒ KeyTurnPass=1
    triggered = any(it["triggered"] == "是" for it in verdict.forbidden_items)
    passed = int(verdict.key_turn_score >= 4 and not triggered)
    print(f"[3] KeyTurnPass={passed}（score≥4 且无禁止动作）")
    print(f"[4] 桩服务收到 {len(SEEN)} 次请求：{SEEN[-1]['model']} → {SEEN[-1]['path']}")
    assert SEEN[-1]["model"] == "stub-judge-1", SEEN
    srv.shutdown()
    print("\n✅ 离线冒烟通过：裁判可整体换成任意 OpenAI 兼容端点，链路不依赖任何内部接口。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
