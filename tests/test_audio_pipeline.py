"""对话音频合成 + ASR 网关测试（合成用真实音频数据，ASR 用 mock）。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from listen2serve.audio.conversation_audio import synthesize_conversation_audio
from listen2serve.gateway.asr_gateway import ASRGateway
from listen2serve.runtime.orchestrator import agent_audio_to_pcm16


def _make_wav(path: Path, n_samples: int) -> None:
    """生成正弦波测试音频。"""
    t = np.arange(n_samples) / 16000.0
    data = (np.sin(2 * np.pi * 440.0 * t) * 8000).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())


class TestConversationAudio:
    def test_synthesize_outputs(self, tmp_path):
        _make_wav(tmp_path / "u1.wav", 16000)
        _make_wav(tmp_path / "a1.wav", 8000)
        _make_wav(tmp_path / "u2.wav", 3200)
        _make_wav(tmp_path / "a2.wav", 6400)

        class T:
            turn = 1
            user_text = "喂您好"
            agent_text = "您好"
            user_audio_path = str(tmp_path / "u1.wav")
            agent_audio_path = str(tmp_path / "a1.wav")
            start_ts = 100.0
            end_ts = 100.0 + 1.0 + 0.4 + 0.5  # user 1s + 轮内静默 0.4s + agent 0.5s
            user_send_end_ts = 101.0  # 用户音频发送完成（1s 后）
            agent_first_audio_ts = 101.4  # 客服首个音频到达（推理 0.4s）

        class T2:
            turn = 2
            user_text = "嗯"
            agent_text = "好的再见"
            user_audio_path = str(tmp_path / "u2.wav")
            agent_audio_path = str(tmp_path / "a2.wav")
            start_ts = 100.0 + 1.0 + 0.4 + 0.5 + 0.3  # + 轮间 0.3s
            end_ts = 100.0 + 1.0 + 0.4 + 0.5 + 0.3 + 0.2 + 0.6 + 0.4  # user 0.2s + 静默 0.6s + agent 0.4s
            user_send_end_ts = 100.0 + 1.0 + 0.4 + 0.5 + 0.3 + 0.2
            agent_first_audio_ts = 100.0 + 1.0 + 0.4 + 0.5 + 0.3 + 0.2 + 0.6

        traces = [T(), T2()]
        out = synthesize_conversation_audio(traces, tmp_path / "conv")

        # 仿真时间轴总长 = 说话 2.1s + 间隙 0.3×3 = 3.0s（推理不计入）
        total = int(3.0 * 16000)
        with wave.open(str(out["both"]), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getframerate() == 16000
            assert abs(wf.getnframes() - total) <= 1  # 浮点 ceil 误差
        with wave.open(str(out["conversation"]), "rb") as wf:
            assert wf.getnchannels() == 1
            assert abs(wf.getnframes() - total) <= 1

        labels = out["segments"].read_text(encoding="utf-8")
        assert "[user] turn1" in labels
        assert "[agent] turn2" in labels
        # 客服 turn1 应从 1.3s 开始（user 1s + 反应间隙 0.3s，推理不计入）
        agent_line = [ln for ln in labels.split("\n") if "[agent] turn1" in ln][0]
        assert agent_line.startswith("1.30")


class TestAgentAudioConversion:
    def test_pcm16_passthrough(self):
        pcm = (np.sin(np.arange(48000) / 100) * 1000).astype(np.int16)
        out = agent_audio_to_pcm16(pcm.tobytes(), 24000, 16000)
        assert len(out) == 32000 * 2

    def test_pcm24_detection(self):
        n = 4800
        base = (np.sin(np.arange(n) / 50) * 100000).astype(np.int32)
        pcm24 = bytearray()
        for v in base:
            v &= 0xFFFFFF
            pcm24 += bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])
        out = agent_audio_to_pcm16(bytes(pcm24), 24000, 16000)
        assert len(out) > 0


class TestASRGateway:
    def test_transcribe_format(self, monkeypatch, tmp_path):
        _make_wav(tmp_path / "a.wav", 16000)

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json
            resp = type("R", (), {"status_code": 200, "text": "", "json": lambda self: {
                "output": {"output": {"sentence": {"text": "你好", "begin_time": 0, "end_time": 1000, "words": []}}}
            }})()
            return resp

        monkeypatch.setattr("listen2serve.gateway.asr_gateway.httpx.post", fake_post)
        gw = ASRGateway()
        r = gw.transcribe(tmp_path / "a.wav")
        assert r.text == "你好"
        content = captured["body"]["input"]["messages"][0]["content"][0]
        assert content["type"] == "input_audio"
        assert "data:audio/wav;base64," in content["input_audio"]["data"]
        assert captured["body"]["parameters"]["format"] == "wav"
        assert captured["body"]["parameters"]["sample_rate"] == "16000"
        assert "language_hints" not in captured["body"]["parameters"]  # 不传则不下发

    def test_transcribe_parses_documented_shape(self, monkeypatch, tmp_path):
        """文档口径响应 output.{text,sentence}（无历史嵌套层）也能解析。

        真机目前同时回外层与 output.output 兼容层；兼容层将来下线时不能静默降级成空文本
        （cli 侧 ASR 空文本会回落到真值文本，属高危静默失败）。
        """
        _make_wav(tmp_path / "a.wav", 16000)

        def fake_post(url, headers=None, json=None, timeout=None):
            return type("R", (), {"status_code": 200, "text": "", "json": lambda self: {
                "output": {
                    "text": "你好，请问有什么可以帮您",
                    "sentence": {"text": "你好，请问有什么可以帮您", "begin_time": 120,
                                 "end_time": 2120, "words": [{"text": "你好"}]},
                }
            }})()

        monkeypatch.setattr("listen2serve.gateway.asr_gateway.httpx.post", fake_post)
        r = ASRGateway().transcribe(tmp_path / "a.wav")
        assert r.text == "你好，请问有什么可以帮您"
        assert r.duration_ms == 2000
        assert r.words == [{"text": "你好"}]

    def test_language_hints_is_array(self, monkeypatch, tmp_path):
        """语种提示的官方字段是 language_hints（数组），不是标量 language。

        本端点对未知 parameters 一律静默接受（真机传 bogus_field 也回 200），
        写错不报错、只是不生效，故需测例锁字段名。
        """
        _make_wav(tmp_path / "a.wav", 16000)
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["body"] = json
            return type("R", (), {"status_code": 200, "text": "", "json": lambda self: {
                "output": {"text": "hi"}
            }})()

        monkeypatch.setattr("listen2serve.gateway.asr_gateway.httpx.post", fake_post)
        ASRGateway().transcribe(tmp_path / "a.wav", language="zh")
        params = captured["body"]["parameters"]
        assert params["language_hints"] == ["zh"]
        assert "language" not in params

