# Video-R1 SFT Annotator

이 폴더는 `Video-R1-COT-165k`를 기반으로, `Qwen/Qwen2.5-VL-72B-Instruct`를 사용해 annotation을 만들고, 최종적으로 현재 레포의 SFT 파이프라인이 바로 읽을 수 있는 `instruction / input / output` 형식의 JSON까지 생성하는 전용 프레임워크다.

핵심 목적은 두 가지다.

1. 비디오 샘플에 대해 답변 스타일을 annotation
- `DIRECT_ANSWER`
- `COT`
- `LONG_COT`

2. 비디오 샘플에 대해 추론 유형을 annotation
- `ABSTRACT`
- `TEMPORAL`
- `SPATIOTEMPORAL`

그리고 마지막에는 이 annotation 결과를 합쳐서, 기존 `GRPO_Video_2/sft/scripts/train_sft.py`가 바로 읽을 수 있는 SFT 학습용 JSON을 만든다.

## 1. 전체 흐름

```text
Video-R1-COT-165k manifest 다운로드
-> video row만 필터링
-> subset 선택 및 샘플링
-> 원본 비디오 다운로드
-> 프레임 추출
-> processed/train.jsonl 생성
-> answer_style annotation
-> reasoning_type annotation
-> 최종 SFT JSON export
```

즉 이 폴더는 이제 단순 annotation 툴이 아니라, `Video-R1-COT-165k -> SFT-ready dataset`까지 끝내는 데이터 생성기다.

## 2. 폴더 구조

```text
video_r1_sft_annotator/
├── README.md
├── requirements.txt
├── configs/
│   ├── answer_style.yaml
│   ├── reasoning_type.yaml
│   └── export_sft.yaml
├── scripts/
│   ├── prepare_video_r1_cot_data.sh
│   ├── annotate_answer_style.sh
│   ├── annotate_reasoning_type.sh
│   └── export_sft_dataset.sh
└── src/video_r1_sft_annotator/
    ├── __init__.py
    ├── annotate.py
    ├── export_sft_dataset.py
    ├── prepare_video_r1_cot.py
    ├── prompts.py
    └── utils.py
```

## 3. 각 파일의 역할

### `prepare_video_r1_cot.py`

`Video-R1-COT-165k.json`을 읽어서 annotation 대상 비디오 샘플을 준비한다.

하는 일:
- manifest 다운로드
- video row만 필터링
- subset 기준 샘플링
- 원본 비디오 다운로드
- 프레임 추출
- `processed/train.jsonl` 생성

중요한 점:
- 이 단계에서 원본 `process`와 `solution`도 같이 보존한다.
- 그래서 나중에 최종 SFT `output`을 만들 수 있다.

### `annotate.py`

준비된 `processed/train.jsonl`을 읽고 annotation을 만든다.

지원 task:
- `answer_style`
- `reasoning_type`

출력은 JSONL이고, 각 row에는 아래 정보가 들어간다.
- question / options / gold answer
- annotation label / reason
- 프레임 경로
- 모델 원문 응답

### `export_sft_dataset.py`

이 스크립트가 최종 마감 단계다.

입력:
- `processed/train.jsonl`
- `outputs/answer_style/annotations.jsonl`
- `outputs/reasoning_type/annotations.jsonl`

출력:
- `outputs/sft/video_r1_cot_sft.json`

이 최종 파일은 기존 AQuA SFT 데이터와 같은 핵심 형식인 `instruction / input / output` JSON list다.

### `prompts.py`

annotation 프롬프트를 정의한다.

현재 제공하는 라벨:
- answer style: `DIRECT_ANSWER`, `COT`, `LONG_COT`
- reasoning type: `ABSTRACT`, `TEMPORAL`, `SPATIOTEMPORAL`

### `utils.py`

공통 유틸이다.

역할:
- JSON / JSONL 입출력
- Hugging Face dataset 다운로드
- archive 압축 해제
- 비디오 탐색
- 프레임 추출
- 질문 / 선택지 / 정답 파싱
- XML 태그 처리
- annotation 응답 JSON 파싱

## 4. 환경 설치

```bash
cd video_r1_sft_annotator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

주의:
- 실제 annotation은 `Qwen2.5-VL-72B-Instruct`를 사용하므로 큰 GPU 메모리가 필요하다.
- `vllm` 기반이라 보통 Linux + CUDA 환경을 전제로 생각하는 것이 맞다.

## 5. 데이터 준비

### 전체 준비

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --download-mode subset-directories
```

### 작은 샘플만 준비

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --sample-ratio 0.01 \
  --seed 42 \
  --download-mode subset-directories \
  --num-frames 8
```

### 일부 subset만 선택

```bash
bash scripts/prepare_video_r1_cot_data.sh \
  --subsets "LLaVA-Video-178K,STAR" \
  --sample-ratio 0.01 \
  --download-mode subset-directories
```

## 6. 데이터 준비 후 생성되는 파일

```text
data/video_r1_cot/raw/Video-R1-COT-165k.json
data/video_r1_cot/processed/train.jsonl
data/video_r1_cot/processed/frames/train/...
```

설명:
- `raw/`: 원본 manifest와 다운로드한 비디오
- `processed/train.jsonl`: annotation 직전 중간 데이터
- `processed/frames/train/...`: 모델 입력용 프레임

## 7. Annotation 실행

### 답변 스타일 annotation

```bash
bash scripts/annotate_answer_style.sh
```

라벨:
- `DIRECT_ANSWER`
- `COT`
- `LONG_COT`

### 추론 유형 annotation

```bash
bash scripts/annotate_reasoning_type.sh
```

라벨:
- `ABSTRACT`
- `TEMPORAL`
- `SPATIOTEMPORAL`

## 8. 최종 SFT 데이터 export

```bash
bash scripts/export_sft_dataset.sh
```

생성 파일:

```text
outputs/sft/video_r1_cot_sft.json
outputs/sft/summary.json
```

이 파일은 현재 메인 레포의 SFT 파이프라인에 바로 넣을 수 있다.

예를 들면 나중에 `GRPO_Video_2/sft/configs/*.yaml`에서 `train_files`로 연결하는 식이다.

## 9. 설정 파일

### `configs/answer_style.yaml`

주요 항목:
- `model_name_or_path`
- `input_path`
- `output_dir`
- `annotation_task`
- `frames_per_sample`
- `temperature`
- `max_completion_tokens`

### `configs/reasoning_type.yaml`

구조는 거의 같고 `annotation_task`만 다르다.

### `configs/export_sft.yaml`

주요 항목:
- `processed_input_path`
- `answer_style_path`
- `reasoning_type_path`
- `output_path`
- `default_answer_style`
- `default_reasoning_type`

`default_*` 값은 annotation이 없는 row가 있을 때의 fallback이다.

## 10. 최종 출력 형식

### annotation 중간 출력

예:

```text
outputs/answer_style/annotations.jsonl
outputs/reasoning_type/annotations.jsonl
```

이건 중간 산출물이다.

### 최종 SFT 출력

최종 JSON list의 각 row는 대략 이런 구조다.

```json
{
  "instruction": "You are given sampled frames from a video and a question about the video.\nAnswer the question based on the visual evidence.\nPrimary reasoning type: TEMPORAL.\nAnswer style: COT. Use concise multi-step reasoning before the final answer.\n\nQuestion: ...",
  "input": "",
  "output": "<COT>\n...\n</COT>\n<ANSWER>\nB\n</ANSWER>",
  "question_id": "xxx",
  "video_id": "yyy",
  "answer_style": "COT",
  "reasoning_type": "TEMPORAL"
}
```

즉 핵심 3필드는:
- `instruction`
- `input`
- `output`

이고, 나머지는 추적용 메타데이터다.

## 11. 이 출력이 기존 SFT와 어떻게 연결되는가

현재 메인 레포의 SFT는 AQuA 데이터를 아래 형식으로 읽는다.

```json
{
  "instruction": "...",
  "input": "",
  "output": "<COT>...</COT><ANSWER>...</ANSWER>"
}
```

`video_r1_sft_annotator`의 최종 출력도 같은 구조로 맞췄다.

즉 차이는 이거다.
- AQuA: 처음부터 SFT용 텍스트 데이터
- Video-R1 annotator 출력: 비디오 기반 annotation을 거쳐 만든 SFT용 데이터

## 12. 가장 짧은 실행 순서

```bash
cd video_r1_sft_annotator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/prepare_video_r1_cot_data.sh \
  --sample-ratio 0.01 \
  --download-mode subset-directories \
  --num-frames 8

bash scripts/annotate_answer_style.sh
bash scripts/annotate_reasoning_type.sh
bash scripts/export_sft_dataset.sh
```

최종 결과:

```text
outputs/sft/video_r1_cot_sft.json
```

## 13. 한 줄 요약

이 폴더는

```text
Video-R1-COT-165k를 받아서,
프레임 기반 annotation을 수행하고,
최종적으로 SFT 가능한 JSON 데이터셋까지 만드는 전용 파이프라인
```

이라고 이해하면 된다.
