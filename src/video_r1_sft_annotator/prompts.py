from typing import Any


ANSWER_STYLE_LABELS = ("DIRECT_ANSWER", "COT", "LONG_COT")
REASONING_TYPE_LABELS = ("ABSTRACT", "TEMPORAL", "SPATIOTEMPORAL")

'''
여기서 두 가지 Style의 Annotation Prompt를 정의합니다.

- Answer Style: 답변의 형태를 정의합니다.
- Reasoning Type: 답변의 이유를 정의합니다.

Answer Style:
- DIRECT_ANSWER: 답변은 직접 관찰 또는 단순한 단계로 주어집니다.
- COT: 답변은 짧은 여러 단계의 추론으로 주어집니다.
- LONG_COT: 답변은 긴 여러 단계의 추론으로 주어집니다.

Reasoning Type:
- ABSTRACT: 추상적인 이유를 정의합니다.
- TEMPORAL: 시간적인 이유를 정의합니다.
- SPATIOTEMPORAL: 공간적인 이유를 정의합니다.
'''

def build_annotation_prompt(task: str, question: str, options: list[str], gold_answer: str) -> str:
    option_block = "\n".join(options) if options else ""

    if task == "answer_style":
        return (
            "You are annotating the reasoning style required to answer a video question.\n\n"
            "Choose exactly one label:\n"
            "- DIRECT_ANSWER: the answer can be given from direct observation or a single trivial step.\n"
            "- COT: the answer requires short multi-step reasoning over the video.\n"
            "- LONG_COT: the answer requires extended reasoning, synthesis across multiple cues, or deeper chain-of-thought.\n\n"
            "Return strict JSON with keys `label` and `reason`.\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{option_block}\n\n"
            f"Gold answer:\n{gold_answer}\n"
        )

    if task == "reasoning_type":
        return (
            "You are annotating the dominant reasoning type required by a video question.\n\n"
            "Choose exactly one label:\n"
            "- ABSTRACT: semantics, intent, category, commonsense, or non-temporal reasoning is primary.\n"
            "- TEMPORAL: ordering, duration, before/after, progression, or time-dependent reasoning is primary.\n"
            "- SPATIOTEMPORAL: both motion/spatial relations and temporal evolution are central.\n\n"
            "Return strict JSON with keys `label` and `reason`.\n"
            f"Question:\n{question}\n\n"
            f"Options:\n{option_block}\n\n"
            f"Gold answer:\n{gold_answer}\n"
        )

    raise ValueError(f"Unsupported annotation task: {task}")


def system_prompt(task: str) -> str:
    if task == "answer_style":
        labels = ", ".join(ANSWER_STYLE_LABELS)
    elif task == "reasoning_type":
        labels = ", ".join(REASONING_TYPE_LABELS)
    else:
        raise ValueError(f"Unsupported annotation task: {task}")

    return (
        "You are a careful video-annotation assistant.\n"
        "Look at the provided video frames, question, options, and gold answer.\n"
        f"Choose exactly one label from: {labels}.\n"
        "Respond with valid JSON only. Example: {\"label\": \"...\", \"reason\": \"...\"}"
    )
