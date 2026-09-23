"""evaluate 增量落盘 / 断点续评 / 并发调度层。

职责边界（铁律）：
- 只负责调度：逐场景结果原子追加 partial、指纹校验续评、并发编排、429 退避、进度口径；
- 裁判判定逻辑与 prompt 不动；最终产物 verdicts_*.json 格式与 内部方案 前逐字段一致
  （partial 行的调度附加键以 `_` 前缀存储，汇总导出时剔除）；
- no-fallback：退避耗尽 / 非限流失败一律抛错，不静默降级。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PartialRejected(RuntimeError):
    """partial 指纹不一致，拒绝复用（需 --fresh 显式重评）。"""


@dataclass
class EvalTask:
    """单场景评测任务（调度层透传 payload，judge 语义由 judge_fn 负责）。"""

    scenario_id: str
    expected_meta: dict[str, Any] # 本次运行该场景应持有的 judge_meta 指纹
    payload: Any = None


@dataclass
class JudgeOutcome:
    """单场景判定产出：verdict（含 judge_meta）+ 分段耗时（秒）。"""

    verdict: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


def partial_path(reports_dir: str | Path, modality: str) -> Path:
    """partial 文件路径（按模态隔离，防 audio/text 互串）。"""
    return Path(reports_dir) / f"verdicts_partial_{modality}.jsonl"


class PartialStore:
    """逐场景 JSONL 增量追加（进程内锁 + 单行写入后 fsync，中断不撕裂已落行）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, line: dict[str, Any]) -> None:
        data = json.dumps(line, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno()) # NAS 写入时序防护：确保单行真实落盘

    def load(self) -> tuple[dict[str, dict[str, Any]], int]:
        """逐行读入；返回 ({scenario_id: 行}, 损坏行数)。

        尾行被中断撕裂（JSON 不完整）按损坏行跳过，对应场景重评即可，
        不影响其余已落行复用。"""
        records: dict[str, dict[str, Any]] = {}
        corrupted = 0
        if not self.path.exists():
            return records, corrupted
        with open(self.path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                    sid = row.get("scenario_id") if isinstance(row, dict) else None
                    if not sid:
                        corrupted += 1
                        continue
                    records[str(sid)] = row
                except json.JSONDecodeError:
                    corrupted += 1
        return records, corrupted


class WorkerLocalPool:
    """线程局部资源池：每个 worker 懒加载独立实例（裁判实例计数器线程不安全，
    禁止共享）；全部实例登记在册，供结束后聚合统计。"""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._local = threading.local()
        self._lock = threading.Lock()
        self.instances: list[Any] = []

    def get(self) -> Any:
        inst = getattr(self._local, "instance", None)
        if inst is None:
            inst = self._factory()
            self._local.instance = inst
            with self._lock:
                self.instances.append(inst)
        return inst


def is_rate_limited(exc: BaseException) -> bool:
    """识别限流类错误（openai.RateLimitError / httpx 路径的 HTTP 429 RuntimeError 等）。"""
    name = type(exc).__name__
    if "RateLimit" in name:
        return True
    msg = str(exc)
    if "429" in name or "429" in msg:
        return True
    low = msg.lower()
    return any(k in low for k in ("rate limit", "rate_limit", "throttling", "too many requests"))


def backoff_delays(start_s: float = 5.0, cap_s: float = 60.0, max_retries: int = 5) -> list[float]:
    """指数退避序列：起步 5s 倍增、封顶 60s、默认最多 5 次。"""
    delays: list[float] = []
    d = start_s
    for _ in range(max_retries):
        delays.append(d)
        d = min(d * 2, cap_s)
    return delays


def judge_with_backoff(
    judge_fn: Callable[[EvalTask], JudgeOutcome],
    task: EvalTask,
    log: Callable[[str], None],
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = 5,
    backoff_start: float = 5.0,
    backoff_cap: float = 60.0,
) -> JudgeOutcome:
    """单场景判定 + 限流指数退避重试；非限流错误直接抛，退避耗尽按现状抛错。"""
    delays = backoff_delays(backoff_start, backoff_cap, max_retries)
    attempt = 0
    while True:
        try:
            return judge_fn(task)
        except Exception as exc: # 仅拦截限流类，其余原样上抛
            if not is_rate_limited(exc) or attempt >= len(delays):
                raise
            wait = delays[attempt]
            attempt += 1
            log(
                f"⚠ 限流退避: {task.scenario_id} 第 {attempt}/{len(delays)} 次，"
                f"{wait:.0f}s 后重试（{type(exc).__name__}: {str(exc)[:120]}）"
            )
            sleep_fn(wait)


def strip_scheduling_keys(verdict_line: dict[str, Any]) -> dict[str, Any]:
    """剔除 partial 行的调度附加键（`_` 前缀），还原与现状逐字段一致的 verdict。"""
    return {k: v for k, v in verdict_line.items() if not str(k).startswith("_")}


def run_incremental(
    tasks: list[EvalTask],
    judge_fn: Callable[[EvalTask], JudgeOutcome],
    store: PartialStore,
    concurrency: int = 1,
    fresh: bool = False,
    log: Callable[[str], None] = print,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = 5,
) -> list[dict[str, Any]]:
    """编排：断点续评（指纹校验）→ 逐场景判定（可并发）→ 增量落盘 + 进度输出。

    返回按 tasks 原始顺序排列的最终 verdict 列表（调度键已剔除）。
    """
    total = len(tasks)
    expected = {t.scenario_id: t.expected_meta for t in tasks}

    # ---- 断点续评：指纹一致才复用，任一不一致整份拒绝 ----
    reused: dict[str, dict[str, Any]] = {}
    records, corrupted = store.load() if not fresh else ({}, 0)
    if records:
        mismatches: list[str] = []
        extra = 0
        for sid, row in records.items():
            if sid not in expected:
                extra += 1
                continue
            if row.get("judge_meta") != expected[sid]:
                mismatches.append(sid)
            else:
                reused[sid] = row
        if mismatches:
            detail = "、".join(sorted(mismatches)[:5]) + ("…" if len(mismatches) > 5 else "")
            raise PartialRejected(
                f"partial 指纹不一致（judge prompt hash / 裁判模型与本次运行不同）: "
                f"{detail} 等 {len(mismatches)} 条；拒绝复用旧判定。"
                "如确认要全量重评，请加 --fresh 显式重评。"
            )
        log(
            f"断点续评: 复用已落盘判定 {len(reused)}/{total}"
            + (f"；损坏行跳过 {corrupted}" if corrupted else "")
            + (f"；忽略非本次范围场景 {extra}" if extra else "")
        )

    pending = [t for t in tasks if t.scenario_id not in reused]
    if not pending:
        log("全部场景已有可复用判定，直接汇总。")
    elif fresh:
        log(f"--fresh: 忽略既有 partial，全量重评 {len(pending)} 条。")

    # ---- 逐场景判定（并发可配）+ 增量落盘 + 进度 ----
    new_verdicts: dict[str, dict[str, Any]] = {}
    completed = len(reused)
    new_done = 0
    t0 = time.monotonic()

    def process(task: EvalTask) -> JudgeOutcome:
        return judge_with_backoff(
            judge_fn, task, log=log, sleep_fn=sleep_fn, max_retries=max_retries
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(process, t): t for t in pending}
        try:
            for fut in as_completed(futures):
                task = futures[fut]
                outcome = fut.result() # 非限流失败原样上抛（partial 已保留既有进度）
                line = {**outcome.verdict, "_elapsed": outcome.timings}
                store.append(line)
                new_verdicts[task.scenario_id] = outcome.verdict
                completed += 1
                new_done += 1
                elapsed_min = (time.monotonic() - t0) / 60
                remain = total - completed
                throughput = new_done / max(time.monotonic() - t0, 1e-9)
                eta_min = (remain / throughput) / 60 if throughput > 0 else 0.0
                parts = [f"{name}={sec:.1f}s" for name, sec in outcome.timings.items()]
                log(
                    f"[{completed}/{total}] {task.scenario_id} "
                    + " ".join(parts)
                    + f" 累计{elapsed_min:.1f}分 预计剩余{eta_min:.1f}分"
                )
        finally:
            # 失败快速中止：不再等待未开始的场景（进行中的由 worker 自行收尾）
            executor.shutdown(wait=False, cancel_futures=True)

    # ---- 按任务原始顺序汇总（与 内部方案 前遍历顺序逐字节一致）----
    verdicts: list[dict[str, Any]] = []
    for t in tasks:
        if t.scenario_id in reused:
            verdicts.append(strip_scheduling_keys(reused[t.scenario_id]))
        else:
            verdicts.append(new_verdicts[t.scenario_id])
    return verdicts
