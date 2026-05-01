"""
Offline evaluation entrypoint (decoupled from model inference).

Module roles:
- ``core``: parse ``<Decision>/<Reply>``, valid-sample checks, path grouping helpers
- ``semantic``: optional reranker-based semantic matching and metric aggregation
- ``evaluate`` (this module): load benchmark manifest + predictions, align rows, write stats JSON

CLI prints at the end: absolute path of ``--out``, then overall accuracy before/after semantic
matching. Full breakdowns (by folder, domain, etc.) are written only to that stats JSON.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from .core import (
    extract_domain_name,
    extract_folder_name,
    extract_reply_for_semantic,
    infer_decision_xml_or_plain,
    is_correct_by_ground_truth_format_aware,
    is_correct_xml_or_legacy,
    is_valid_result_item,
    load_jsonl,
    raw_model_text,
)
from .semantic import (
    SemanticEvalConfig,
    build_stats_report,
    evaluate_metrics_with_policy,
    try_load_semantic_bundle,
    write_stats_json,
)


def _index_predictions_by_id(pred_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in pred_rows:
        rid = r.get("id")
        if rid is None:
            continue
        out[str(rid)] = r
    return out


def build_aligned_rows(
    *, benchmark_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Join predictions to the benchmark manifest and return rows ready for metrics.

    - The benchmark manifest defines the evaluated sample set (one row per id).
    - Paths for grouping come from the manifest; predictions supply raw text via ``_raw_reply``
      or ``reply`` (see ``raw_model_text``).
    """
    pred_by_id = _index_predictions_by_id(pred_rows)
    aligned: list[dict[str, Any]] = []

    for sample in benchmark_rows:
        gid = str(sample.get("id", ""))
        audio_path = str(sample.get("path", ""))
        gt = sample.get("decision")

        pred = pred_by_id.get(gid, {})
        raw = raw_model_text(pred)
        decision = infer_decision_xml_or_plain(raw)
        correct = bool(gt and is_correct_xml_or_legacy(str(gt), raw))

        aligned.append(
            {
                "id": gid,
                "path": audio_path,
                "folder": extract_folder_name(audio_path),
                "domain": extract_domain_name(audio_path),
                "decision": decision,
                "reply": pred.get("reply", ""),
                "_raw_reply": raw,
                "ground_truth": gt,
                "is_correct": correct,
            }
        )

    return aligned


def evaluate_predictions(
    *,
    benchmark_jsonl: str,
    predictions_jsonl: str,
    output_stats_json: str,
    checkpoint_label: str = "",
    semantic_config: SemanticEvalConfig | None = None,
) -> dict[str, Any]:
    benchmark_rows = load_jsonl(benchmark_jsonl)
    pred_rows = load_jsonl(predictions_jsonl)

    results = build_aligned_rows(benchmark_rows=benchmark_rows, pred_rows=pred_rows)
    valid_results = [r for r in results if is_valid_result_item(r)]

    semantic_available, standard_map, reranker, semantic_error, threshold = try_load_semantic_bundle(
        semantic_config
    )
    legacy_metrics = evaluate_metrics_with_policy(
        valid_results=valid_results,
        semantic_available=semantic_available,
        standard_map=standard_map,
        reranker=reranker,
        semantic_threshold=threshold,
        decision_getter=lambda item: infer_decision_xml_or_plain(raw_model_text(item)),
        correctness_checker=is_correct_by_ground_truth_format_aware,
        semantic_query_getter=extract_reply_for_semantic,
    )

    report = build_stats_report(
        checkpoint=checkpoint_label,
        semantic_config=semantic_config,
        semantic_available=semantic_available,
        semantic_error=semantic_error,
        legacy_metrics=legacy_metrics,
    )
    write_stats_json(output_stats_json, report)
    return report


def _cli() -> None:
    p = argparse.ArgumentParser(description="Evaluate proactive benchmark predictions (JSONL).")
    p.add_argument(
        "--benchmark",
        required=True,
        help="benchmark manifest jsonl (e.g. proactive_reply_dataset_v3.jsonl)",
    )
    p.add_argument("--pred", required=True, help="predictions jsonl (must contain id + model raw output)")
    p.add_argument("--out", required=True, help="output stats json path")
    p.add_argument("--checkpoint-label", default="", help="checkpoint label written into stats")

    p.add_argument("--respond-jsonl", default="", help="respond standard answers jsonl (optional)")
    p.add_argument("--reranker-dir", default="", help="FlagEmbedding reranker model dir (optional)")
    p.add_argument("--semantic-threshold", type=float, default=0.5)

    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    semantic_config = None
    if args.respond_jsonl and args.reranker_dir:
        semantic_config = SemanticEvalConfig(
            standard_jsonl=args.respond_jsonl,
            reranker_model_path=args.reranker_dir,
            threshold=args.semantic_threshold,
        )

    report = evaluate_predictions(
        benchmark_jsonl=args.benchmark,
        predictions_jsonl=args.pred,
        output_stats_json=args.out,
        checkpoint_label=args.checkpoint_label,
        semantic_config=semantic_config,
    )

    overall_before = float(report["before_semantic"]["overall"]["accuracy"])
    overall_after = float(report["after_semantic"]["overall"]["accuracy"])
    print(f"Stats written to: {os.path.abspath(args.out)}")
    print(f"Overall accuracy before semantic matching (before_semantic): {overall_before:.2%}")
    print(f"Overall accuracy after semantic matching (after_semantic): {overall_after:.2%}")


if __name__ == "__main__":
    _cli()

