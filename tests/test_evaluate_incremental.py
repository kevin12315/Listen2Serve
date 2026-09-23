"""Plan H evaluate 增量落盘/断点续评/并发调度层单测（fake judge，不打真实 API）。

覆盖：partial 恢复、指纹不一致拒绝复用、--fresh 全量重评、并发下 verdicts 完整性
与 worker 独立实例计数器聚合、429 指数退避路径（成功/耗尽/非限流不重试）。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from listen2serve.evaluation.incremental import (
    EvalTask,
    JudgeOutcome,
    PartialRejected,
    PartialStore,
    WorkerLocalPool,
    backoff_delays,
    is_rate_limited,
    judge_with_backoff,
    partial_path,
    run_incremental,
    strip_scheduling_keys,
)


def _meta(version: str = "v2.1-B8", model: str = "stub-judge-model") -> dict:
    """构造与 build_judge_meta 同构的指纹（测试内自洽即可）。"""
    return {
        "policy": {"version": version, "prompt_hash": "ph_policy", "model": model},
        "voice": {"version": version, "prompt_hash": "ph_voice", "model": model},
        "dialogue": None,
    }


def _tasks(n: int, meta: dict | None = None) -> list[EvalTask]:
    return [
        EvalTask(scenario_id=f"S{i:02d}", expected_meta=meta or _meta(), payload=i)
        for i in range(n)
    ]


def _make_judge_fn(calls: list[str], meta: dict | None = None):
    def judge_fn(task: EvalTask) -> JudgeOutcome:
        calls.append(task.scenario_id)
        return JudgeOutcome(
            verdict={
                "scenario_id": task.scenario_id,
                "policy_pass": True,
                "judge_meta": meta or _meta(),
            },
            timings={"policy": 1.0, "voice": 2.0},
        )

    return judge_fn


class TestPartialStore:
    def test_append_load_roundtrip(self, tmp_path):
        store = PartialStore(tmp_path / "verdicts_partial_audio.jsonl")
        store.append({"scenario_id": "S01", "policy_pass": True})
        store.append({"scenario_id": "S02", "policy_pass": False})
        records, corrupted = store.load()
        assert corrupted == 0
        assert set(records) == {"S01", "S02"}
        assert records["S02"]["policy_pass"] is False

    def test_load_tolerates_corrupt_line(self, tmp_path):
        path = tmp_path / "verdicts_partial_audio.jsonl"
        store = PartialStore(path)
        store.append({"scenario_id": "S01", "judge_meta": _meta()})
        # 模拟中断撕裂的尾行（JSON 不完整）
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"scenario_id": "S02", "judge_me\n')
        records, corrupted = store.load()
        assert corrupted == 1
        assert set(records) == {"S01"}  # 损坏行跳过，其余可复用

    def test_partial_path_isolated_by_modality(self, tmp_path):
        assert partial_path(tmp_path, "audio").name == "verdicts_partial_audio.jsonl"
        assert partial_path(tmp_path, "text").name == "verdicts_partial_text.jsonl"


class TestResume:
    def test_resume_skips_matching_fingerprint(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "policy_pass": True, "judge_meta": _meta()})
        calls: list[str] = []
        verdicts = run_incremental(
            _tasks(3), _make_judge_fn(calls), store,
            log=lambda msg: None, sleep_fn=lambda s: None,
        )
        assert calls == ["S01", "S02"]  # S00 指纹一致 → 跳过
        assert [v["scenario_id"] for v in verdicts] == ["S00", "S01", "S02"]  # 顺序还原

    def test_fingerprint_mismatch_rejects_whole_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        # 旧判定用的是另一裁判模型 → 指纹不一致
        store.append({"scenario_id": "S00", "policy_pass": True,
                      "judge_meta": _meta(model="qwen3-max")})
        with pytest.raises(PartialRejected, match="--fresh"):
            run_incremental(_tasks(3), _make_judge_fn([]), store,
                            log=lambda msg: None, sleep_fn=lambda s: None)

    def test_fresh_ignores_mismatched_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "policy_pass": True,
                      "judge_meta": _meta(model="qwen3-max")})
        calls: list[str] = []
        verdicts = run_incremental(
            _tasks(2), _make_judge_fn(calls), store, fresh=True,
            log=lambda msg: None, sleep_fn=lambda s: None,
        )
        assert calls == ["S00", "S01"]  # --fresh 全量重评，不校验不复用
        assert len(verdicts) == 2

    def test_strips_scheduling_keys_on_reuse(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S00", "judge_meta": _meta(), "_elapsed": {"policy": 1.0}})
        verdicts = run_incremental(
            _tasks(1), _make_judge_fn([]), store,
            log=lambda msg: None, sleep_fn=lambda s: None,
        )
        assert "_elapsed" not in verdicts[0]
        assert strip_scheduling_keys({"a": 1, "_elapsed": {}}) == {"a": 1}


class TestConcurrency:
    def test_concurrent_completeness_order_and_partial(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        calls: list[str] = []
        tasks = _tasks(8)
        verdicts = run_incremental(
            tasks, _make_judge_fn(calls), store, concurrency=4,
            log=lambda msg: None, sleep_fn=lambda s: None,
        )
        assert sorted(calls) == sorted(t.scenario_id for t in tasks)
        assert [v["scenario_id"] for v in verdicts] == [t.scenario_id for t in tasks]
        records, corrupted = store.load()
        assert corrupted == 0 and len(records) == 8
        # partial 行携带 _elapsed 调度键（审计用），最终产物剔除
        assert "_elapsed" in records["S00"]

    def test_worker_local_pool_isolated_instances_and_aggregation(self):
        """每 worker 独立实例（实例计数器线程不安全）+ 按 worker 聚合。"""

        class FakeJudge:
            def __init__(self) -> None:
                self.parse_retries = 0

        pool = WorkerLocalPool(FakeJudge)
        seen: list[FakeJudge] = []
        lock = threading.Lock()

        def work(_: int) -> None:
            inst = pool.get()
            inst.parse_retries += 1  # 各 worker 只写自己的实例
            with lock:
                seen.append(inst)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(work, range(16)))
        # 独立实例数 == worker 数（≤4），聚合后计数总和 == 任务总数
        assert len(pool.instances) <= 4
        assert len(set(map(id, pool.instances))) == len(pool.instances)
        assert sum(i.parse_retries for i in pool.instances) == 16
        assert set(seen) <= set(pool.instances)


class TestRateLimitBackoff:
    def test_backoff_delays_sequence(self):
        assert backoff_delays() == [5.0, 10.0, 20.0, 40.0, 60.0]

    def test_is_rate_limited(self):
        assert is_rate_limited(RuntimeError("DashScope omni 评分失败 HTTP 429: xxx"))
        assert is_rate_limited(RuntimeError("Error code: 429 - rate limit exceeded"))

        class RateLimitError(Exception):
            pass

        assert is_rate_limited(RateLimitError("too many requests"))
        assert is_rate_limited(RuntimeError("Throttling: quota exceeded"))
        assert not is_rate_limited(ValueError("JSON parse error"))

    def test_retry_then_success(self):
        sleeps: list[float] = []
        n = {"calls": 0}

        def flaky(task: EvalTask) -> JudgeOutcome:
            n["calls"] += 1
            if n["calls"] <= 2:
                raise RuntimeError("HTTP 429: rate limited")
            return JudgeOutcome(verdict={"scenario_id": task.scenario_id})

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        out = judge_with_backoff(flaky, task, log=lambda msg: None,
                                 sleep_fn=sleeps.append)
        assert out.verdict["scenario_id"] == "S01"
        assert sleeps == [5.0, 10.0]  # 起步 5s 指数退避

    def test_backoff_exhausted_raises(self):
        sleeps: list[float] = []

        def always_429(task: EvalTask) -> JudgeOutcome:
            raise RuntimeError("HTTP 429: rate limited")

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        with pytest.raises(RuntimeError, match="429"):
            judge_with_backoff(always_429, task, log=lambda msg: None,
                               sleep_fn=sleeps.append)
        assert len(sleeps) == 5 and sleeps[-1] == 60.0  # 最多 5 次、封顶 60s

    def test_non_rate_limit_error_not_retried(self):
        sleeps: list[float] = []
        n = {"calls": 0}

        def boom(task: EvalTask) -> JudgeOutcome:
            n["calls"] += 1
            raise ValueError("judge 调用失败")

        task = EvalTask(scenario_id="S01", expected_meta=_meta())
        with pytest.raises(ValueError):
            judge_with_backoff(boom, task, log=lambda msg: None,
                               sleep_fn=sleeps.append)
        assert n["calls"] == 1 and sleeps == []


class TestPartialLineFormat:
    def test_partial_line_json_serializable_single_line(self, tmp_path):
        store = PartialStore(tmp_path / "p.jsonl")
        store.append({"scenario_id": "S01", "judge_meta": _meta(),
                      "policy_detail": {"reason": "中文理由"}, "_elapsed": {"policy": 1.5}})
        lines = store.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["policy_detail"]["reason"] == "中文理由"
