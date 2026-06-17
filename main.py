import argparse
import json
import os
from pathlib import Path

from zina import DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL_ID, ZinaModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_ID))
    parser.add_argument("--processor", default=os.environ.get("PROCESSOR_PATH"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--cand", required=True)
    parser.add_argument("--refs", default=None)
    parser.add_argument("--ref", action="append", default=None)
    parser.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--torch-dtype", default=os.environ.get("TORCH_DTYPE", "auto"))
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)),
    )
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_image(value):
    if value.startswith(("http://", "https://", "file://")):
        return value
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"image not found: {value}")
    return str(path.resolve())


def resolve_refs(args):
    if args.ref:
        return args.ref
    if args.refs is None:
        return ""
    try:
        parsed = json.loads(args.refs)
    except json.JSONDecodeError:
        return args.refs
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return str(parsed)


def main():
    args = parse_args()
    model = ZinaModel(
        model_path=args.model,
        processor_path=args.processor,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        flash_attn=args.flash_attn,
        verbose=args.verbose,
    )
    pred_with_tags, pred = model.run(
        cand=args.cand,
        refs=resolve_refs(args),
        image_path=resolve_image(args.image),
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "pred_with_tags": pred_with_tags,
                "pred": pred,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
