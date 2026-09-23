"""客观指标（复用 τ 思路自研）：DB 终态比对、action 序列匹配、pass^k。

不依赖 LLM，全部确定性计算。
定位（v5.8，2026-08-14）：**弃用预留**——评测默认不发起 function call
（Settings.tools_enabled=False），db_terminal_reward/action_sequence_match 不再产生
输入；机制代码完整保留（开关打开可复活），不再列入指标演进计划。
注：pass_hat_k 不依赖工具，语义不变（需多 trial 数据，）。
"""

from __future__ import annotations

from math import comb
from typing import Any


def db_terminal_reward(db: Any, expectations: dict[str, list[tuple]]) -> tuple[float, dict[str, Any]]:
    """DB 终态比对：返回 (reward 0-1, 明细)。

    Args:
        db: sqlite 连接
        expectations: {table: [conds, ...]}，每个 conds 是 [(column, expected_value), ...]，
            表示期望表中存在满足全部条件的行。

    Returns:
        reward = 命中的期望条数 / 总期望条数（无期望时为 1.0）
    """
    matched = 0
    total = 0
    details: dict[str, Any] = {}
    for table, conds_list in expectations.items():
        table_matched = 0
        for conds in conds_list:
            where = " AND ".join(f"{c} = ?" for c, _ in conds)
            row = db.execute(
                f"SELECT 1 FROM {table} WHERE {where} LIMIT 1", # noqa: S608 - 列名来自评测定义
                [v for _, v in conds],
            ).fetchone()
            if row is not None:
                table_matched += 1
        matched += table_matched
        total += len(conds_list)
        details[table] = {"expected": len(conds_list), "matched": table_matched}
    reward = matched / total if total else 1.0
    return reward, details


def action_sequence_match(
    actions: list[tuple[str, dict[str, Any]]],
    gold_sequence: list[tuple[str, dict[str, Any] | None]],
) -> tuple[float, list[bool]]:
    """action 序列比对：按顺序匹配（gold 中的 None 表示不校验参数）。

    注意：与 τ2 的口径不同——τ2 无序匹配且全有全无（0/1）；本实现要求顺序、
    给部分分（gold 命中数 / gold 总数），适配"服务动作推进"类评测。

    Returns:
        (匹配率, 每步是否匹配)
    """
    hits: list[bool] = []
    gold_idx = 0
    for name, args in actions:
        if gold_idx >= len(gold_sequence):
            break
        gold_name, gold_args = gold_sequence[gold_idx]
        if name != gold_name:
            hits.append(False)
            continue
        if gold_args is None or all(args.get(k) == v for k, v in gold_args.items()):
            hits.append(True)
            gold_idx += 1
        else:
            hits.append(False)
    if not gold_sequence:
        return 1.0, hits
    return gold_idx / len(gold_sequence), hits


def all_pass(conversation_ok: bool, db_ok: bool, action_ok: bool) -> bool:
    """单次运行的联合通过判定（各维度 AND，τ2 的 reward 乘法组合同语义）。"""
    return conversation_ok and db_ok and action_ok


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """pass^k（τ2 定义）：num_trials 次试验中随机抽 k 次全部成功的概率。

    公式：C(success_count, k) / C(num_trials, k)

    Args:
        num_trials: 同一任务的总试验次数 n
        success_count: 其中成功次数 c
        k: 抽取次数（k ≤ n）
    """
    if k <= 0 or num_trials <= 0 or k > num_trials:
        raise ValueError(f"非法参数: num_trials={num_trials}, success_count={success_count}, k={k}")
    if success_count < k:
        return 0.0
    return comb(success_count, k) / comb(num_trials, k)


def aggregate_pass_hat_k(
    trials_by_task: dict[str, list[bool]], k: int
) -> tuple[float, dict[str, float]]:
    """多任务 pass^k 聚合：逐任务计算后取平均（τ2 agent_metrics 同语义）。

    Args:
        trials_by_task: {task_id: [每次试验是否成功]}
        k: 抽取次数；试验数不足 k 的任务跳过

    Returns:
        (平均 pass^k, 逐任务明细)
    """
    per_task: dict[str, float] = {}
    for task_id, trials in trials_by_task.items():
        n = len(trials)
        if n < k:
            continue
        per_task[task_id] = pass_hat_k(n, sum(1 for t in trials if t), k)
    mean = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return mean, per_task
