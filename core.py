import json
import re
from typing import Any

RAW_MODEL_TEXT_KEY = "_raw_reply"
"""Full raw model output field key (optional in saved JSON). Choose SYSTEM_PROMPT yourself."""
SYSTEM_PROMPT = ("")

USER_TASK_TEXT = """You are a proactively helpful model.
Your task is to determine whether to assist the user based on the provided audio.
- If the audio indicates physiological danger or environmental safety risks, provide appropriate assistance.
- If it reflects negative emotions, offer mental comfort.
- If it suggests equipment malfunction, provide warnings or safety guidance.
- If none of the above apply,just respond with "No Reply".Do not explain anything.
Your final output must strictly follow this format:
<Decision>(RESPOND or IGNORE)</Decision>
<Reply>(Your reply)</Reply>"""


def extract_folder_name(path: str) -> str:
    match = re.search(r"/sound_v3/([^/]+/[^/]+)/", path)
    if match:
        return match.group(1)
    return "Unknown"


def extract_domain_name(path: str) -> str:
    match = re.search(r"/sound_v3/([^/]+)/", path)
    if match:
        return match.group(1)
    return "Unknown"


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_tag_content(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = rf"<\s*{tag}\s*>(.*?)<\s*/\s*{tag}\s*>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def parse_formatted_model_reply(reply_text: str) -> tuple[str, str]:
    raw = reply_text or ""
    return _extract_tag_content(raw, "Decision"), _extract_tag_content(raw, "Reply")


def reply_body_for_record(full_text: str) -> str:
    _, inner = parse_formatted_model_reply(full_text or "")
    return inner


def raw_model_text(item: dict[str, Any]) -> str:
    return (item.get(RAW_MODEL_TEXT_KEY) or item.get("reply") or "").strip()


def result_dict_for_jsonl(result: dict[str, Any], *, save_raw_reply: bool) -> dict[str, Any]:
    if save_raw_reply:
        return dict(result)
    return {k: v for k, v in result.items() if k != RAW_MODEL_TEXT_KEY}


def _decision_from_xml_inner(decision_inner: str) -> str:
    if not decision_inner:
        return ""
    match = re.search(r"\b(RESPOND|IGNORE)\b", decision_inner, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def infer_decision_xml_or_plain(reply_text: str) -> str:
    """Parse RESPOND or IGNORE only from a closed <Decision>...</Decision> block; else return ''."""
    decision_inner, _ = parse_formatted_model_reply(reply_text or "")
    parsed = _decision_from_xml_inner(decision_inner)
    return parsed if parsed in {"RESPOND", "IGNORE"} else ""


def extract_reply_for_semantic(reply_text: str) -> str:
    """Semantic scoring uses only the inner text of <Reply>...</Reply>; empty if missing."""
    _, reply_inner = parse_formatted_model_reply(reply_text or "")
    return reply_inner


def is_correct_xml_or_legacy(ground_truth: str, reply: str) -> bool:
    """Correct iff parsed <Decision> label matches ground_truth RESPOND/IGNORE (no plain-text fallback)."""
    gt = (ground_truth or "").strip().upper()
    if gt not in {"RESPOND", "IGNORE"}:
        return False
    decision_inner, _ = parse_formatted_model_reply(reply or "")
    parsed_dec = _decision_from_xml_inner(decision_inner)
    if parsed_dec not in {"RESPOND", "IGNORE"}:
        return False
    return parsed_dec == gt


def is_correct_by_ground_truth_format_aware(ground_truth: str, decision: str, reply: str) -> bool:
    return is_correct_xml_or_legacy(ground_truth, reply)


def is_valid_result_item(item: dict[str, Any]) -> bool:
    ground_truth = (item.get("ground_truth") or "").upper()
    decision = (item.get("decision") or "").upper()
    reply = raw_model_text(item)
    if ground_truth not in {"RESPOND", "IGNORE"}:
        return False
    if decision not in {"RESPOND", "IGNORE"}:
        return False
    if reply.startswith("Error:"):
        return False
    return True


def summarize_dataset_domains(items: list[dict[str, Any]]) -> None:
    from collections import defaultdict

    folder_data: dict[str, list] = defaultdict(list)
    domain_data: dict[str, list] = defaultdict(list)
    for item in items:
        p = item.get("path", "")
        folder_data[extract_folder_name(str(p))].append(item)
        domain_data[extract_domain_name(str(p))].append(item)

    print("\nCounts by top-level domain:")
    for domain_name, data_list in sorted(domain_data.items()):
        print(f"  {domain_name}: {len(data_list)} samples")

    print("\nCounts by domain/subcategory (folder):")
    total_count = 0
    for folder_name, data_list in sorted(folder_data.items()):
        count = len(data_list)
        total_count += count
        print(f"  {folder_name}: {count} samples")
    print(f"  Total: {total_count} samples")
