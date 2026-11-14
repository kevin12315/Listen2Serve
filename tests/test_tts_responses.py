"""The TTS adapter accepts its documented binary and URL response forms."""

import io
import wave

import httpx
import pytest

from listen2serve.gateway.tts_gateway import TTSGateway, _extract_audio_bytes
from listen2serve.runtime.config import Settings


def wav_bytes():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x01\x00" * 160)
    return stream.getvalue()


def test_binary_wav_does_not_attempt_json_decoding(monkeypatch):
    data = wav_bytes()
    monkeypatch.setattr("httpx.post", lambda *a, **k: httpx.Response(200, content=data))
    output = TTSGateway(Settings(_env_file=None, dashscope_api_key="fixture")).synthesize("你好")
    assert output.audio == b"\x01\x00" * 160
    assert output.sample_rate == 16000


def test_audio_url_is_downloaded_instead_of_base64_decoded(monkeypatch):
    seen = []

    def download(url, **kwargs):
        seen.append(url)
        return httpx.Response(200, content=wav_bytes(), request=httpx.Request("GET", url))

    monkeypatch.setattr("httpx.get", download)
    result = _extract_audio_bytes({"output": {"audio_url": "https://example.test/clip.wav"}}, b"json")
    assert seen == ["https://example.test/clip.wav"]
    assert result.startswith(b"RIFF")


def test_missing_audio_is_an_error():
    with pytest.raises(RuntimeError, match="no audio"):
        _extract_audio_bytes({"output": {}}, b"{}")
