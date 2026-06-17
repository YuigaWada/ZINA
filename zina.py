import logging
import re
import torch
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


DEFAULT_MODEL_ID = "yuwd/zina-72b"
DEFAULT_MAX_NEW_TOKENS = 2048
PROJECT_ROOT = Path(__file__).resolve().parent


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",  # White
        logging.INFO: "\033[36m",  # Cyan
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredFormatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEBUG = False


class ZinaModel:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_ID,
        processor_path=None,
        verbose=False,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        flash_attn=False,
    ):
        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.verbose = verbose
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.flash_attn = flash_attn
        self._load_model()

    def run(self, cand, refs, image_path, verbose=False):
        if verbose:
            logger.info("Performing detection")
        detected = self.detect(cand, refs, image_path)

        if verbose:
            logger.info("Performing editing")

        pred_with_tags, pred = self.edit(cand, refs, image_path, detected)
        return pred_with_tags, pred

    def detect(self, cand, refs, image_path):
        prompt = self._build_prompts(cand, refs, "detect")
        res = self._qwen_inference(prompt, image_path)
        try:
            # three, number / apple, object / ...
            splitted = res.split("/")  # [(target,tag),...]
            detected = []
            for x in splitted:
                if len(x.split(",")) == 2:
                    target, tag = x.split(",")
                    detected.append((target.strip(), tag.strip()))
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            detected = []

        return detected

    def edit(self, cand, refs, image_path, detected):
        target_list = []
        for target, tag_type in detected:
            locs = [i for i in range(len(cand)) if cand.startswith(target, i)]
            t = [(loc, target, tag_type) for loc in locs]
            target_list.extend(t)

        target_list_ = []
        seen = set()
        for item in target_list:
            loc, target, x = item
            key = f"{loc}_{target}_{x}"
            if key not in seen:
                target_list_.append(item)
                seen.add(key)

        target_list = target_list_

        def inference(cap_copy):
            if DEBUG:
                return "truck"
            else:
                prompt = self._build_prompts(cap_copy, refs, "edit")
                res = self._qwen_inference(prompt, image_path)
                if len(res.split(":")) != 2:
                    return None

                orig, edited = [x.strip() for x in res.split(":")]
                if orig.startswith("<"):
                    orig = orig.split(">")[1].split("<")[0]
                    orig = orig.strip()
                return edited

        target_list.sort()
        pred_with_tags = deepcopy(cand)
        pred = deepcopy(cand)

        count = defaultdict(int)
        for _, target, tag_type in target_list:
            cap_copy = deepcopy(pred)
            if target not in cap_copy:
                continue

            count[target] += 1
            tag = self._make_tags(target, tag_type)
            cap_copy = self._replace_ith_occurance(count[target], cap_copy, target, tag)
            if "<" not in cap_copy:
                continue

            edited = inference(cap_copy)
            if edited is None:
                edited = target

            edited, target = [x.strip() for x in [edited, target]]
            if edited == target:  # reject
                pass
            else:  # accept
                tags_with_edited = self._make_tags(edited, tag_type)
                pred = self._replace_ith_occurance(count[target], pred, target, edited)
                pred_with_tags = self._replace_ith_occurance(
                    count[target], pred_with_tags, target, tags_with_edited
                )
                count[target] -= 1

        return pred_with_tags, pred

    def _load_model(self):
        kwargs = {}
        if self.flash_attn:
            kwargs = {"attn_implementation": "flash_attention_2"}

        logger.info(f"Loading model: {self.model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            **kwargs,
        )

        logger.info(f"Loading processor: {self.processor_path}")
        processor = AutoProcessor.from_pretrained(self.processor_path)
        self.model = model
        self.processor = processor
        self.input_device = self._infer_input_device(model)

    def _infer_input_device(self, model):
        if torch.cuda.is_available():
            for parameter in model.parameters():
                if parameter.device.type == "cuda":
                    return parameter.device
            return torch.device("cuda")
        for parameter in model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        return torch.device("cpu")

    def _qwen_inference(self, prompt, image):
        model, processor = self.model, self.processor
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.input_device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if isinstance(output_text, list):
            output_text = output_text[0]

        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_text

    def _build_prompts(self, cand, refs, mode):
        prompt_path = ""
        if mode == "detect":
            prompt_path = PROJECT_ROOT / "prompts" / "prompt_detect.md"
        elif mode == "edit":
            prompt_path = PROJECT_ROOT / "prompts" / "prompt_edit.md"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        prompts = "".join(prompts)
        prompts = prompts.replace("[Original]", cand)
        prompts = prompts.replace("[Reference]", self._format_refs(refs))
        return prompts

    def _format_refs(self, refs):
        if isinstance(refs, str):
            return refs
        if refs is None:
            return ""
        return "\n".join(str(ref) for ref in refs)

    def _replace_ith_occurance(self, index, sentences, target, replacement):
        pattern = rf"(?<!(?:[A-Za-z><])){re.escape(target)}(?![A-Za-z><])"
        occurrences = [m.start() for m in re.finditer(pattern, sentences)]
        if 1 <= index <= len(occurrences):
            start = occurrences[index - 1]
            end = start + len(target)
            return sentences[:start] + replacement + sentences[end:]
        else:
            print(f"Failed {target} --> {replacement}")
            print(
                f"Index {index} is out of bounds (found {len(occurrences)} occurrences). Returning original string."
            )
            return sentences

    def _make_tags(self, target, tag_type):
        return f"<{tag_type}>{target}</{tag_type}>"
