import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from .prompts import ANSWER_STYLE_LABELS, REASONING_TYPE_LABELS
from .utils import (
    build_question_with_options,
    extract_xml_tag_text,
    load_jsonl,
    normalize_answer_text,
    strip_outer_reasoning_tag,
    write_json,
)


@dataclass
class ExportConfig:
    output_path: str
    summary_path: str
    # --- classic annotation-based export ---
    processed_input_path: Optional[str] = None
    answer_style_path: Optional[str] = None
    reasoning_type_path: Optional[str] = None
    include_reasoning_type_hint: bool = True
    include_answer_style_hint: bool = True
    default_answer_style: str = "COT"
    default_reasoning_type: str = "ABSTRACT"
    # --- generation-based export (generate_sft task) ---
    generated_path: Optional[str] = None
    max_samples: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export annotated Video-R1-COT rows to SFT-ready JSON.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> ExportConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExportConfig(**raw)


def sample_key(row: dict) -> str:
    question_id = str(row.get("question_id") or "").strip()
    video_id = str(row.get("video_id") or "").strip()
    if question_id:
        return question_id
    return f"{video_id}::{str(row.get('question') or '').strip()}"


def load_annotation_map(path: Path, expected_task: str, allowed_labels: set[str]) -> dict[str, dict]:
    rows = load_jsonl(path)
    indexed: dict[str, dict] = {}
    for row in rows:
        if str(row.get("annotation_task") or "").strip() != expected_task:
            continue
        label = str(row.get("annotation_label") or "").strip().upper()
        if label not in allowed_labels:
            continue
        indexed[sample_key(row)] = row
    return indexed


def normalize_reasoning_trace(process_raw: str) -> str:
    text = extract_xml_tag_text(process_raw, "think") or strip_outer_reasoning_tag(process_raw)
    return text.strip()


def build_output(answer_style: str, reasoning_trace: str, answer_text: str) -> str:
    answer_block = f"<ANSWER>\n{answer_text}\n</ANSWER>"
    if answer_style == "DIRECT_ANSWER":
        return answer_block
    if answer_style == "COT":
        cot_body = reasoning_trace or answer_text
        return f"<COT>\n{cot_body}\n</COT>\n{answer_block}"
    if answer_style == "LONG_COT":
        cot_body = reasoning_trace or answer_text
        return f"<LONG_COT>\n{cot_body}\n</LONG_COT>\n{answer_block}"
    raise ValueError(f"Unsupported answer style: {answer_style}")


def build_instruction(
    question: str,
    options: list[str],
    reasoning_type: str,
    answer_style: str,
    include_reasoning_type_hint: bool,
    include_answer_style_hint: bool,
) -> str:
    lines = [
        "You are given sampled frames from a video and a question about the video.",
        "Answer the question based on the visual evidence.",
    ]
    if include_reasoning_type_hint:
        lines.append(f"Primary reasoning type: {reasoning_type}.")
    if include_answer_style_hint:
        if answer_style == "DIRECT_ANSWER":
            lines.append("Answer style: DIRECT_ANSWER. Respond directly with the final answer.")
        elif answer_style == "COT":
            lines.append("Answer style: COT. Use concise multi-step reasoning before the final answer.")
        elif answer_style == "LONG_COT":
            lines.append("Answer style: LONG_COT. Use detailed multi-step reasoning before the final answer.")
    lines.append("")
    lines.append(build_question_with_options(question=question, options=options))
    return "\n".join(line for line in lines if line is not None).strip()


def export_from_generated(
    generated_rows: list[dict],
    max_samples: Optional[int],
) -> tuple[list[dict], dict]:
    if max_samples is not None:
        generated_rows = generated_rows[:max_samples]

    exported_rows: list[dict] = []
    stats: Counter = Counter()

    for row in generated_rows:
        question = str(row.get("question") or "")
        options = [str(opt) for opt in (row.get("options") or [])]
        instruction = build_question_with_options(question=question, options=options)

        answer_raw = str(row.get("answer_raw") or "").strip()
        cot_raw = str(row.get("cot_raw") or "").strip()
        long_cot_raw = str(row.get("long_cot_raw") or "").strip()

        if not answer_raw:
            stats["skip_missing_answer"] += 1
            continue

        base = {
            "input": "",
            "question_id": row.get("question_id"),
            "video_id": row.get("video_id"),
            "source_subset": row.get("source_subset"),
            "question_category": row.get("question_category"),
            "frames": row.get("frames"),
            "gold_answer": answer_raw,
        }

        exported_rows.append({
            "instruction": instruction,
            "output": f"<ANSWER>\n{answer_raw}\n</ANSWER>",
            "reasoning_depth": "ANSWER",
            **base,
        })
        stats["ANSWER"] += 1

        if cot_raw:
            exported_rows.append({
                "instruction": instruction,
                "output": f"<COT>\n{cot_raw}\n</COT>\n<ANSWER>\n{answer_raw}\n</ANSWER>",
                "reasoning_depth": "COT",
                **base,
            })
            stats["COT"] += 1
        else:
            stats["skip_missing_cot"] += 1

        if long_cot_raw:
            exported_rows.append({
                "instruction": instruction,
                "output": f"<LONG_COT>\n{long_cot_raw}\n</LONG_COT>\n<ANSWER>\n{answer_raw}\n</ANSWER>",
                "reasoning_depth": "LONG_COT",
                **base,
            })
            stats["LONG_COT"] += 1
        else:
            stats["skip_missing_long_cot"] += 1

    return exported_rows, dict(stats)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- generation-based export (generate_sft task) ---
    if cfg.generated_path:
        generated_rows = load_jsonl(Path(cfg.generated_path))
        if not generated_rows:
            raise SystemExit(f"No rows found in {cfg.generated_path}")
        exported_rows, stats = export_from_generated(generated_rows, cfg.max_samples)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(exported_rows, f, ensure_ascii=False, indent=2)
        summary = {
            "generated_path": str(Path(cfg.generated_path).resolve()),
            "output_path": str(output_path.resolve()),
            "input_rows": len(generated_rows),
            "exported_rows": len(exported_rows),
            "stats": stats,
        }
        write_json(Path(cfg.summary_path), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # --- classic annotation-based export ---
    if not cfg.processed_input_path:
        raise SystemExit("Either 'generated_path' or 'processed_input_path' must be set in config.")

    processed_rows = load_jsonl(Path(cfg.processed_input_path))
    if cfg.max_samples is not None:
        processed_rows = processed_rows[: cfg.max_samples]
    if not processed_rows:
        raise SystemExit(f"No processed rows found in {cfg.processed_input_path}")

    answer_style_map = load_annotation_map(
        path=Path(cfg.answer_style_path or ""),
        expected_task="answer_style",
        allowed_labels=set(ANSWER_STYLE_LABELS),
    )
    reasoning_type_map = load_annotation_map(
        path=Path(cfg.reasoning_type_path or ""),
        expected_task="reasoning_type",
        allowed_labels=set(REASONING_TYPE_LABELS),
    )

    default_answer_style = cfg.default_answer_style.strip().upper()
    default_reasoning_type = cfg.default_reasoning_type.strip().upper()
    if default_answer_style not in set(ANSWER_STYLE_LABELS):
        raise ValueError(f"Unsupported default_answer_style: {cfg.default_answer_style}")
    if default_reasoning_type not in set(REASONING_TYPE_LABELS):
        raise ValueError(f"Unsupported default_reasoning_type: {cfg.default_reasoning_type}")

    stats: Counter = Counter()
    label_counts: Counter = Counter()
    exported_rows = []

    for row in processed_rows:
        key = sample_key(row)
        answer_style = str(answer_style_map.get(key, {}).get("annotation_label") or default_answer_style).strip().upper()
        reasoning_type = str(reasoning_type_map.get(key, {}).get("annotation_label") or default_reasoning_type).strip().upper()

        answer_text = normalize_answer_text(str(row.get("solution_raw") or row.get("answer") or ""))
        if not answer_text:
            answer_text = normalize_answer_text(str(row.get("answer") or ""))
        if not answer_text:
            stats["skip_missing_answer"] += 1
            continue

        reasoning_trace = normalize_reasoning_trace(str(row.get("process_raw") or ""))
        if answer_style != "DIRECT_ANSWER" and not reasoning_trace:
            stats["skip_missing_reasoning_trace"] += 1
            continue

        instruction = build_instruction(
            question=str(row.get("question") or ""),
            options=[str(opt) for opt in (row.get("options") or [])],
            reasoning_type=reasoning_type,
            answer_style=answer_style,
            include_reasoning_type_hint=cfg.include_reasoning_type_hint,
            include_answer_style_hint=cfg.include_answer_style_hint,
        )
        output = build_output(
            answer_style=answer_style,
            reasoning_trace=reasoning_trace,
            answer_text=answer_text,
        )

        exported_rows.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
                "question_id": row.get("question_id"),
                "video_id": row.get("video_id"),
                "source_subset": row.get("source_subset"),
                "question_category": row.get("question_category"),
                "frames": row.get("frame_subdir"),
                "answer_style": answer_style,
                "reasoning_type": reasoning_type,
                "gold_answer": answer_text,
            }
        )
        label_counts[f"answer_style:{answer_style}"] += 1
        label_counts[f"reasoning_type:{reasoning_type}"] += 1
        stats["exported_rows"] += 1
        if key in answer_style_map:
            stats["rows_with_answer_style_annotation"] += 1
        else:
            stats["rows_with_default_answer_style"] += 1
        if key in reasoning_type_map:
            stats["rows_with_reasoning_type_annotation"] += 1
        else:
            stats["rows_with_default_reasoning_type"] += 1

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(exported_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "processed_input_path": str(Path(cfg.processed_input_path).resolve()),
        "answer_style_path": str(Path(cfg.answer_style_path or "").resolve()),
        "reasoning_type_path": str(Path(cfg.reasoning_type_path or "").resolve()),
        "output_path": str(output_path.resolve()),
        "input_rows": len(processed_rows),
        "exported_rows": len(exported_rows),
        "stats": dict(stats),
        "label_counts": dict(label_counts),
    }
    write_json(Path(cfg.summary_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
