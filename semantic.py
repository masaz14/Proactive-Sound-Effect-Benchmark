"""Semantic Matching Metric: Based on the list of standard answers and Reranker scores."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from .core import extract_domain_name, load_jsonl, raw_model_text


@dataclass
class SemanticEvalConfig:
    standard_jsonl: str
    reranker_model_path: str
    threshold: float = 0.5


def build_standard_map(standard_data: list[dict[str, Any]]) -> dict[str, list[str]]:
    standard_map: dict[str, list[str]] = {}
    for item in standard_data:
        item_id = item.get("id")
        answers = item.get("standard_answers", [])
        if item_id and isinstance(answers, list) and answers:
            standard_map[str(item_id)] = [str(a) for a in answers]
    return standard_map


def load_reranker(model_path: str):
    import torch

    try:
        from FlagEmbedding import FlagReranker
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "FlagEmbedding is not installed; semantic matching cannot be performed. Run: pip install FlagEmbedding."
        ) from e

    use_fp16 = torch.cuda.is_available()
    try:
        return FlagReranker(model_path, use_fp16=use_fp16)
    except Exception:
        if use_fp16:
            return FlagReranker(model_path, use_fp16=False)
        raise


def calculate_similarities_batch(
    reranker, query: str, documents: list[str], *, normalize: bool = True
) -> list[float]:
    pairs = [[query, doc] for doc in documents]
    scores = reranker.compute_score(pairs, normalize=normalize)
    return [float(score) for score in scores]


def evaluate_metrics_with_policy(
    valid_results: list[dict[str, Any]],
    *,
    semantic_available: bool,
    standard_map: dict[str, list[str]],
    reranker,
    semantic_threshold: float,
    decision_getter: Callable[[dict[str, Any]], str],
    correctness_checker: Callable[[str, str, str], bool],
    semantic_query_getter: Callable[[str], str],
) -> dict[str, Any]:
    before_folder_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    after_folder_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    before_domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    after_domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    gt_respond_total = 0
    gt_respond_before_correct = 0
    gt_respond_after_correct = 0

    for item in valid_results:
        folder = item.get("folder", "Unknown")
        domain = item.get("domain") or extract_domain_name(str(item.get("path", "")))
        gt = (item.get("ground_truth") or "").upper()
        reply_text = raw_model_text(item)
        item_id = str(item.get("id", ""))
        decision = decision_getter(item)

        if gt == "IGNORE":
            before_correct = correctness_checker(gt, decision, reply_text)
            after_correct = before_correct
        elif gt == "RESPOND":
            if decision != "RESPOND":
                before_correct = False
                after_correct = False
            else:
                before_correct = True
                after_correct = False
        else:
            before_correct = False
            after_correct = False

        before_folder_stats[folder]["total"] += 1
        if before_correct:
            before_folder_stats[folder]["correct"] += 1
        before_domain_stats[domain]["total"] += 1
        if before_correct:
            before_domain_stats[domain]["correct"] += 1

        if gt == "RESPOND":
            gt_respond_total += 1
            if before_correct:
                gt_respond_before_correct += 1

        if semantic_available and gt == "RESPOND" and decision == "RESPOND":
            standard_answers = standard_map.get(item_id, [])
            semantic_query = semantic_query_getter(reply_text)
            if standard_answers and semantic_query:
                sims = calculate_similarities_batch(
                    reranker, semantic_query, standard_answers, normalize=True
                )
                max_sim = max(sims) if sims else 0.0
                if max_sim > semantic_threshold:
                    after_correct = True

        if not semantic_available:
            after_correct = before_correct

        after_folder_stats[folder]["total"] += 1
        if after_correct:
            after_folder_stats[folder]["correct"] += 1
        after_domain_stats[domain]["total"] += 1
        if after_correct:
            after_domain_stats[domain]["correct"] += 1

        if gt == "RESPOND" and after_correct:
            gt_respond_after_correct += 1

    def _build_folder_report(folder_stats):
        report: dict[str, Any] = {}
        total_all = 0
        correct_all = 0
        for folder_name in sorted(folder_stats.keys()):
            stats = folder_stats[folder_name]
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            total_all += total
            correct_all += correct
            report[folder_name] = {
                "total_samples": total,
                "correct_samples": correct,
                "accuracy": acc,
            }
        overall = (correct_all / total_all) if total_all > 0 else 0.0
        return report, overall, {"total_samples": total_all, "correct_samples": correct_all}

    by_folder_before, overall_by_folder_before, overall_counts_before = _build_folder_report(
        before_folder_stats
    )
    by_folder_after, overall_by_folder_after, overall_counts_after = _build_folder_report(
        after_folder_stats
    )
    by_domain_before, _, _ = _build_folder_report(before_domain_stats)
    by_domain_after, _, _ = _build_folder_report(after_domain_stats)

    gt_respond_before_accuracy = (
        (gt_respond_before_correct / gt_respond_total) if gt_respond_total > 0 else 0.0
    )
    gt_respond_after_accuracy = (
        (gt_respond_after_correct / gt_respond_total) if gt_respond_total > 0 else 0.0
    )

    return {
        "before_semantic": {
            "overall": {
                "total_samples": overall_counts_before["total_samples"],
                "correct_samples": overall_counts_before["correct_samples"],
                "accuracy": overall_by_folder_before,
            },
            "ground_truth_respond_accuracy": gt_respond_before_accuracy,
            "by_folder": by_folder_before,
            "by_domain": by_domain_before,
        },
        "after_semantic": {
            "overall": {
                "total_samples": overall_counts_after["total_samples"],
                "correct_samples": overall_counts_after["correct_samples"],
                "accuracy": overall_by_folder_after,
            },
            "ground_truth_respond_accuracy": gt_respond_after_accuracy,
            "by_folder": by_folder_after,
            "by_domain": by_domain_after,
        },
    }


def try_load_semantic_bundle(
    config: SemanticEvalConfig | None,
) -> tuple[bool, dict[str, list[str]], Any, str, float]:
    if config is None:
        return False, {}, None, "", 0.5
    threshold = float(config.threshold)
    if not os.path.exists(config.standard_jsonl) or not os.path.exists(config.reranker_model_path):
        return False, {}, None, "The semantic matching standard file or reranker path does not exist.", threshold
    try:
        standard_map = build_standard_map(load_jsonl(config.standard_jsonl))
        reranker = load_reranker(config.reranker_model_path)
        return True, standard_map, reranker, "", threshold
    except Exception as e:
        return False, {}, None, str(e), threshold


def build_stats_report(
    *,
    checkpoint: str,
    semantic_config: SemanticEvalConfig | None,
    semantic_available: bool,
    semantic_error: str,
    legacy_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "semantic_config": {
            "standard_file": getattr(semantic_config, "standard_jsonl", "") if semantic_config else "",
            "model_path": getattr(semantic_config, "reranker_model_path", "")
            if semantic_config
            else "",
            "threshold": getattr(semantic_config, "threshold", 0.5) if semantic_config else 0.5,
            "semantic_available": semantic_available,
            "semantic_error": semantic_error,
        },
        "before_semantic": legacy_metrics["before_semantic"],
        "after_semantic": legacy_metrics["after_semantic"],
    }


def write_stats_json(path: str, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
