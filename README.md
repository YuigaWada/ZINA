# ZINA: Multimodal Fine-grained Hallucination Detection and Editing

> Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we construct VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and Llama-3.2, in both detection and editing tasks.

https://arxiv.org/abs/2506.13130

## Installation

Dependencies are managed with `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

FlashAttention can be installed separately if needed:

```bash
uv pip install wheel ninja packaging setuptools
uv pip install flash_attn --no-build-isolation
```

## Quick Start

Run inference with an image, a generated caption, and a reference caption:

```bash
uv run python main.py \
  --image /path/to/image.jpg \
  --cand "A blue bus is at the station." \
  --ref "A red train is at the station."
```

The command returns JSON. `pred_with_tags` preserves ZINA's span annotations, while `pred` contains the edited caption without tags.

```json
{
  "model": "...",
  "pred_with_tags": "...",
  "pred": "..."
}
```
