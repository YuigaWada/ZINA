import json
import os
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from numbers import Number
from pathlib import Path

import numpy as np


DEFAULT_BERT_F1_MODEL = "microsoft/deberta-xlarge-mnli"
DEFAULT_BERT_F1_NUM_LAYERS = 40
DEFAULT_CLIP_F1_MODEL = "ViT-B/32"
WORD_LEVEL_METRICS = {"f1", "soft-f1", "bert", "clip"}
WORD_LEVEL_METRIC_ALIASES = {
    "bert-f1": "bert",
    "bertf1": "bert",
    "clip-f1": "clip",
    "clipf1": "clip",
}

CAPEVALKIT_METRIC_ALIASES = {
    "bleu": "bleu",
    "bleu-1": "bleu",
    "bleu-2": "bleu",
    "bleu-3": "bleu",
    "bleu-4": "bleu",
    "bleu1": "bleu",
    "bleu2": "bleu",
    "bleu3": "bleu",
    "bleu4": "bleu",
    "meteor": "meteor",
    "rouge": "rouge",
    "rouge-l": "rouge",
    "rougel": "rouge",
    "cider": "cider",
    "ciderd": "cider",
    "spice": "spice",
    "clip": "clipscore",
    "clip-score": "clipscore",
    "clipscore": "clipscore",
    "clipscore-vitl": "clipscore-vitl",
    "clipscorevitl": "clipscore-vitl",
    "clipscoreavg": "clipscoreavg",
    "clipscore-average": "clipscoreavg",
    "clipscoreaverage": "clipscoreavg",
    "refclipscore": "refclipscore",
    "refclip-score": "refclipscore",
    "refclipscore-vitl": "refclipscore-vitl",
    "refclipscorevitl": "refclipscore-vitl",
    "pac": "pacscore",
    "pacs": "pacscore",
    "pac-s": "pacscore",
    "pacscore": "pacscore",
    "pacscore-vitl": "pacscore-vitl",
    "pacscorevitl": "pacscore-vitl",
    "pac-s-vitl": "pacscore-vitl",
    "pacs-vitl": "pacscore-vitl",
    "pacscoreavg": "pacscoreavg",
    "pacsavg": "pacscoreavg",
    "pac-savg": "pacscoreavg",
    "pacscore-average": "pacscoreavg",
    "pacscoreaverage": "pacscoreavg",
    "refpac": "refpacscore",
    "refpacs": "refpacscore",
    "refpac-s": "refpacscore",
    "refpacscore": "refpacscore",
    "refpacscore-vitl": "refpacscore-vitl",
    "refpacscorevitl": "refpacscore-vitl",
    "refpac-s-vitl": "refpacscore-vitl",
    "refpacs-vitl": "refpacscore-vitl",
    "pac-s++": "pacscorepp",
    "pacs++": "pacscorepp",
    "pacspp": "pacscorepp",
    "pacscore++": "pacscorepp",
    "pacscorepp": "pacscorepp",
    "pac-s++avg": "pacscoreppavg",
    "pacs++avg": "pacscoreppavg",
    "pacsppavg": "pacscoreppavg",
    "pacscore++avg": "pacscoreppavg",
    "pacscoreppavg": "pacscoreppavg",
    "refpac-s++": "refpacscorepp",
    "refpacs++": "refpacscorepp",
    "refpacspp": "refpacscorepp",
    "refpacscore++": "refpacscorepp",
    "refpacscorepp": "refpacscorepp",
    "fleur": "fleur",
    "reffleur": "reffleur",
    "polos": "polos",
    "vela": "vela",
}


def normalize_capevalkit_metric_name(metric):
    name = str(metric).strip().lower().replace("_", "-").replace(" ", "-")
    compact = name.replace("-", "")
    if name in CAPEVALKIT_METRIC_ALIASES:
        return CAPEVALKIT_METRIC_ALIASES[name]
    if compact in CAPEVALKIT_METRIC_ALIASES:
        return CAPEVALKIT_METRIC_ALIASES[compact]
    return name


def normalize_word_metric_name(metric):
    name = str(metric).strip().lower().replace("_", "-").replace(" ", "-")
    compact = name.replace("-", "")
    if name in WORD_LEVEL_METRIC_ALIASES:
        return WORD_LEVEL_METRIC_ALIASES[name]
    if compact in WORD_LEVEL_METRIC_ALIASES:
        return WORD_LEVEL_METRIC_ALIASES[compact]
    return name


def evaluate_captions_with_capevalkit(**kwargs):
    from capevalkit import evaluate_captions

    return evaluate_captions(**kwargs)


@dataclass
class EvalTarget:
    pred: str
    cand: str
    refs: list
    image_path: str
    pred_with_tags: str
    labels_with_tags: str


class HallucinationEvalTool:
    def __init__(
        self,
        target_model_name="test",
        models=None,
        verbose=False,
        bert_model_name=None,
        bert_num_layers=None,
        clip_model_name=None,
        embedding_device=None,
        capevalkit_output_dir=None,
    ):
        self.target_model_name = target_model_name
        self.verbose = verbose
        self.bert_model_name = bert_model_name or os.environ.get(
            "ZINA_BERT_F1_MODEL", DEFAULT_BERT_F1_MODEL
        )
        self.bert_num_layers = self._resolve_bert_num_layers(bert_num_layers)
        self.clip_model_name = clip_model_name or os.environ.get(
            "ZINA_CLIP_F1_MODEL", DEFAULT_CLIP_F1_MODEL
        )
        self.embedding_device = embedding_device or os.environ.get(
            "ZINA_EVAL_DEVICE"
        )
        self.capevalkit_output_dir = capevalkit_output_dir
        self._embedding_models = {}
        self._bert_score_loaded = False
        self.default_caption_metrics = self._caption_metrics_from_models(models)
        self.pycoco_eval_cap_scorers = []

    @staticmethod
    def _resolve_bert_num_layers(value):
        if value is None:
            value = os.environ.get("ZINA_BERT_F1_NUM_LAYERS")
        if value is None:
            return DEFAULT_BERT_F1_NUM_LAYERS
        if value == "":
            return None
        return int(value)

    def convert(self, cand, pred_with_tags, labels_with_tags, refs, image_path):
        pred = self.remove_tags(pred_with_tags)
        labels = self.remove_tags(labels_with_tags)
        return EvalTarget(
            pred=pred,
            cand=cand,
            refs=refs + [labels],
            image_path=image_path,
            pred_with_tags=pred_with_tags,
            labels_with_tags=labels_with_tags,
        )

    def run(self, targets, mode="detector", metrics=None, write_results=False):
        if not targets:
            return {}, {}
        if metrics is None:
            metrics = ["f1"]
        assert isinstance(targets[0], EvalTarget)

        word_metrics, caption_metrics = self._split_metrics(metrics)
        caption_metrics = self._unique(
            self.default_caption_metrics + caption_metrics
        )
        word_level_scores = defaultdict(list)
        valid_mask = [True] * len(targets)

        for metric in word_metrics:
            for index, target in enumerate(targets):
                f1_score, precision, recall = self.evaluate_word_level(
                    target.cand,
                    target.pred_with_tags,
                    target.labels_with_tags,
                    mode,
                    metric,
                    index,
                )
                word_level_scores[f"{metric}_f1"].append(f1_score)
                word_level_scores[f"{metric}_precision"].append(precision)
                word_level_scores[f"{metric}_recall"].append(recall)
                valid_mask[index] = valid_mask[index] and f1_score is not None

        all_scores = dict(word_level_scores)
        if caption_metrics:
            _, caption_scores = self.evaluate_sentence_level(
                targets, metrics=caption_metrics
            )
            all_scores.update(caption_scores)

        mean_scores, all_scores = self._finalize_scores(all_scores, valid_mask)

        if write_results:
            self._write_results(targets, all_scores, valid_mask)

        return mean_scores, all_scores

    def _write_results(self, targets, all_scores, valid_mask):
        targets_with_scores = [
            {
                "pred": target.pred,
                "cand": target.cand,
                "references": target.refs,
                "image_path": target.image_path,
                "pred_with_tags": target.pred_with_tags,
                "labels_with_tags": target.labels_with_tags,
            }
            for target in targets
        ]
        targets_with_scores = deepcopy(targets_with_scores)
        for metric_name, scores in all_scores.items():
            for index, score in enumerate(scores):
                targets_with_scores[index][metric_name] = (
                    score if valid_mask[index] else -1
                )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path("results") / Path(
            f"results_{self.target_model_name}_{timestamp}.json"
        )
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(targets_with_scores, f, indent=4)

    @staticmethod
    def extract_tags(text):
        pattern = r"<([^>]+)>([^<]+)</([^>]+)>"
        results = []
        for open_tag, content, end_tag in re.findall(pattern, text):
            original_text = f"<{open_tag}>{content}</{end_tag}>"
            results.append(
                {
                    "open_tag": open_tag,
                    "content": content,
                    "end_tag": end_tag,
                    "original_text": original_text,
                }
            )
        return results

    def _extract_tags(self, text):
        return HallucinationEvalTool.extract_tags(text)

    def remove_tags(self, text):
        for tag in self.extract_tags(text):
            text = text.replace(tag["original_text"], tag["content"])
        return text

    def evaluate_word_level(
        self, original, pred, labels, mode="detector", metric="f1", idx=None
    ):
        metric = normalize_word_metric_name(metric)
        assert mode in ["detector", "editor"], (
            f"mode must be either 'detector' or 'editor' ({mode}?)"
        )

        preprocessed = self._get_mapping_tuple(original, pred, labels)
        pred_mapping = preprocessed["pred_mapping"]
        label_mapping = preprocessed["label_mapping"]
        if pred_mapping is None or label_mapping is None:
            return None, None, None

        original = preprocessed["original"]
        if len(label_mapping) != len(original.split()):
            raise AssertionError("Mismatch in token count between prediction and labels.")

        if metric not in WORD_LEVEL_METRICS:
            raise NotImplementedError(f"Unsupported word-level metric: {metric}")

        is_valid = True
        pred_total = 0
        label_total = 0
        prev_pred_base = None
        prev_label_base = None
        tuple_pred_count = defaultdict(int)
        tuple_label_count = defaultdict(int)
        tuple_sim = {}

        for pred_tuple, label_tuple in zip(pred_mapping, label_mapping):
            pred_before, pred_token, pred_tag = pred_tuple
            _, label_token, label_tag = label_tuple

            if pred_tag == label_tag == "normal" and pred_token != label_token:
                is_valid = False
                break

            if pred_tag != "normal":
                if mode == "detector" and metric == "f1" and pred_token != pred_before:
                    is_valid = False
                    break
                pred_base = pred_token[:-1]
                if prev_pred_base != pred_base:
                    pred_total += 1
                    key = f"{pred_token[:-1]}_{label_token[:-1]}"
                    if metric == "f1" and pred_token == label_token:
                        tuple_pred_count[key] += 1
                        tuple_sim[key] = 1
                    elif metric == "soft-f1" and pred_tag == label_tag:
                        tuple_pred_count[key] += 1
                        tuple_sim[key] = 1
                    elif metric not in ["f1", "soft-f1"]:
                        tuple_pred_count[key] += 1
                        tuple_sim[key] = self._compute_sim(
                            pred_token[:-1], label_token[:-1], metric
                        )
                prev_pred_base = pred_base
            else:
                prev_pred_base = None

            if label_tag != "normal":
                label_base = label_token[:-1]
                if prev_label_base != label_base:
                    label_total += 1
                    key = f"{pred_token[:-1]}_{label_token[:-1]}"
                    if metric == "f1" and pred_token == label_token:
                        tuple_label_count[key] += 1
                        tuple_sim[key] = 1
                    elif metric == "soft-f1" and pred_tag == label_tag:
                        tuple_label_count[key] += 1
                        tuple_sim[key] = 1
                    elif metric not in ["f1", "soft-f1"]:
                        tuple_label_count[key] += 1
                        tuple_sim[key] = self._compute_sim(
                            pred_token[:-1], label_token[:-1], metric
                        )
                prev_label_base = label_base
            else:
                prev_label_base = None

        pred_correct = 0.0
        label_correct = 0.0
        for key, value in tuple_sim.items():
            if key in tuple_pred_count and key in tuple_label_count:
                count = min(tuple_pred_count[key], tuple_label_count[key])
                pred_correct += count * value
                label_correct += count * value

        if not is_valid and pred_total == 0 and label_total == 0:
            return 0.0, 0.0, 0.0

        assert abs(pred_correct - label_correct) < 1e-6, (
            f"model: {metric}, pred_correct: {pred_correct}, label_correct: {label_correct}"
        )
        return self._calculate_scores(pred_correct, pred_total, label_total)

    @staticmethod
    def _caption_metrics_from_models(models):
        if not models:
            return []
        if isinstance(models, str):
            models = [models]
        normalized = {str(model).lower() for model in models}
        metrics = []
        if "pycoco" in normalized or "coco" in normalized:
            metrics.extend(["bleu", "rouge", "cider"])
        if "clip" in normalized:
            metrics.extend(["clipscoreavg", "pacscoreavg"])
        if "capevalkit" in normalized:
            metrics.extend(["bleu", "rouge", "cider", "clipscoreavg", "pacscoreavg"])
        return HallucinationEvalTool._unique(metrics)

    @staticmethod
    def _split_metrics(metrics):
        word_metrics = []
        caption_metrics = []
        for metric in metrics:
            word_name = normalize_word_metric_name(metric)
            if word_name in WORD_LEVEL_METRICS:
                word_metrics.append(word_name)
            else:
                caption_metrics.append(normalize_capevalkit_metric_name(metric))
        return (
            HallucinationEvalTool._unique(word_metrics),
            HallucinationEvalTool._unique(caption_metrics),
        )

    @staticmethod
    def _unique(values):
        unique_values = []
        seen = set()
        for value in values:
            if value in seen:
                continue
            unique_values.append(value)
            seen.add(value)
        return unique_values

    @staticmethod
    def _finalize_scores(all_scores, valid_mask):
        mean_scores = {}
        finalized_scores = {}
        for metric_name, scores in all_scores.items():
            values = []
            finalized = []
            for index, score in enumerate(scores):
                if valid_mask[index] and score is not None:
                    value = float(score)
                    values.append(value)
                    finalized.append(value)
                else:
                    finalized.append(None)
            mean_scores[metric_name] = float(np.mean(values)) if values else 0.0
            finalized_scores[metric_name] = finalized
        return mean_scores, finalized_scores

    def _calculate_scores(self, matched, pred_total, label_total):
        if pred_total == 0 and label_total == 0:
            return 0.0, 0.0, 0.0
        if pred_total == 0:
            precision = 0.0
            recall = matched / label_total
            f1_score = 0.0
        elif label_total == 0:
            precision = matched / pred_total
            recall = 0.0
            f1_score = 0.0
        elif matched == 0:
            precision = 0.0
            recall = 0.0
            f1_score = 0.0
        else:
            precision = matched / pred_total
            recall = matched / label_total
            f1_score = 2 * precision * recall / (precision + recall)

        return tuple(score * 100 for score in [f1_score, precision, recall])

    def evaluate_sentence_level(
        self, targets, use_ptb_tokenizer=False, metrics=None
    ):
        if not targets:
            return {}, {}
        if metrics is None:
            metrics = self.default_caption_metrics
        metric_names = self._unique(
            [normalize_capevalkit_metric_name(metric) for metric in metrics]
        )
        if not metric_names:
            return {}, {}
        pairs = self._targets_to_capevalkit_pairs(targets)
        results = evaluate_captions_with_capevalkit(
            metrics=metric_names,
            pairs=pairs,
            output_dir=self._capevalkit_output_dir(),
            quiet=not self.verbose,
        )
        return self._normalize_capevalkit_results(results, len(targets), scale=100.0)

    def _pycoco_eval(self, name, scorer, refs, cands, ims_cs, gen_cs):
        metrics = [normalize_capevalkit_metric_name(name)]
        pairs = self._pairs_from_metric_inputs(refs, cands, ims_cs, gen_cs)
        results = evaluate_captions_with_capevalkit(
            metrics=metrics,
            pairs=pairs,
            output_dir=self._capevalkit_output_dir(),
            quiet=not self.verbose,
        )
        mean_scores, all_scores = self._normalize_capevalkit_results(
            results, len(pairs), scale=1.0
        )
        if name == "BLEU":
            names = [f"BLEU-{index}" for index in range(1, 5)]
            return [mean_scores[item] for item in names], [all_scores[item] for item in names]
        metric_name = next(iter(all_scores))
        return mean_scores[metric_name], all_scores[metric_name]

    def _get_all_ic_metrics(
        self,
        refs,
        cands,
        ims_cs,
        gen_cs,
        return_per_cap=False,
        spice_need_trancuate=True,
    ):
        pairs = self._pairs_from_metric_inputs(refs, cands, ims_cs, gen_cs)
        metrics = self.default_caption_metrics
        if not pairs or not metrics:
            return {}
        results = evaluate_captions_with_capevalkit(
            metrics=metrics,
            pairs=pairs,
            output_dir=self._capevalkit_output_dir(),
            quiet=not self.verbose,
        )
        mean_scores, all_scores = self._normalize_capevalkit_results(
            results, len(pairs), scale=1.0
        )
        return all_scores if return_per_cap else mean_scores

    @staticmethod
    def _targets_to_capevalkit_pairs(targets):
        pairs = []
        for index, target in enumerate(targets):
            references = target.refs
            if isinstance(references, str):
                references = [references]
            elif references is None:
                references = []
            else:
                references = list(references)
            pairs.append(
                {
                    "id": str(index),
                    "image": target.image_path,
                    "caption": target.pred,
                    "references": references,
                }
            )
        return pairs

    @staticmethod
    def _pairs_from_metric_inputs(refs, cands, ims_cs, gen_cs):
        pairs = []
        ids = list(cands)
        for index, item_id in enumerate(ids):
            candidate = cands[item_id]
            if isinstance(candidate, list):
                candidate = candidate[0]
            references = refs.get(item_id, refs.get(str(index), refs.get(f"{index}_0", [])))
            if isinstance(references, str):
                references = [references]
            pairs.append(
                {
                    "id": str(index),
                    "image": ims_cs[index],
                    "caption": candidate if candidate is not None else gen_cs[index],
                    "references": list(references),
                }
            )
        return pairs

    def _normalize_capevalkit_results(self, results, target_count, scale=100.0):
        mean_scores = {}
        all_scores = {}
        for metric_name, score, per_item in self._iter_capevalkit_outputs(results):
            scores = self._capevalkit_per_item_scores(
                metric_name, per_item, target_count
            )
            score_value = float(score) if score is not None else float(np.mean(scores))
            all_scores[metric_name] = [float(value) * scale for value in scores]
            mean_scores[metric_name] = score_value * scale
        return mean_scores, all_scores

    def _capevalkit_output_dir(self):
        if self.capevalkit_output_dir is not None:
            return self.capevalkit_output_dir
        return Path("results") / "capevalkit" / self.target_model_name

    def _iter_capevalkit_outputs(self, payload, parent_key=None):
        if not isinstance(payload, Mapping):
            return
        per_item = payload.get("per_item")
        if "score" in payload and isinstance(per_item, Mapping):
            yield str(parent_key or payload.get("name") or "score"), payload.get("score"), per_item
            return
        if isinstance(per_item, Mapping):
            for key, value in payload.items():
                if key == "per_item" or not isinstance(value, Number):
                    continue
                yield str(key), value, per_item
            return
        for key, value in payload.items():
            if isinstance(value, Mapping):
                yield from self._iter_capevalkit_outputs(value, str(key))

    @staticmethod
    def _capevalkit_per_item_scores(metric_name, per_item, target_count):
        scores = []
        for index in range(target_count):
            value = per_item.get(str(index), per_item.get(index))
            if isinstance(value, Mapping):
                if metric_name in value:
                    value = value[metric_name]
                elif len(value) == 1:
                    value = next(iter(value.values()))
                else:
                    raise KeyError(f"{metric_name} missing from per-item score {index}")
            if value is None:
                raise KeyError(f"missing per-item score {index} for {metric_name}")
            scores.append(float(value))
        return scores

    def _parse_markup(self, text):
        pattern = re.compile(r"<([^>]+)>(.*?)</\1>")
        tokens = []
        placeholder_mapping = {}
        pos = 0
        markup_index = 0

        for match in pattern.finditer(text):
            plain_text = text[pos : match.start()]
            tokens.extend(plain_text.split())

            tag = match.group(1)
            markup_content = match.group(2).strip()
            placeholder = f"__MARKUP_{markup_index}__"
            markup_index += 1

            tokens.append(placeholder)
            placeholder_mapping[placeholder] = (markup_content, tag)
            pos = match.end()

        plain_text = text[pos:]
        tokens.extend(plain_text.split())
        return tokens, placeholder_mapping

    def _create_token_mapping(self, original, marked_text, is_pred=False):
        orig_tokens = original.split()
        marked_tokens, placeholder_mapping = self._parse_markup(marked_text)
        mapping = []
        orig_index = 0
        token_index = 0

        while token_index < len(marked_tokens):
            current_token = marked_tokens[token_index]
            if not current_token.startswith("__MARKUP_"):
                if orig_index >= len(orig_tokens):
                    if not is_pred:
                        return None
                    break

                mapping.append((orig_tokens[orig_index], current_token, "normal"))
                orig_index += 1
                token_index += 1
                continue

            markup_group = []
            group_tags = []
            while token_index < len(marked_tokens) and marked_tokens[
                token_index
            ].startswith("__MARKUP_"):
                placeholder = marked_tokens[token_index]
                markup_group.append(placeholder)
                tag = placeholder_mapping[placeholder][1]
                group_tags.append(tag)
                token_index += 1

            next_plain = (
                marked_tokens[token_index] if token_index < len(marked_tokens) else None
            )
            if next_plain is not None:
                boundary = orig_index + 1
                while boundary < len(orig_tokens) and orig_tokens[boundary] != next_plain:
                    boundary += 1
            else:
                boundary = len(orig_tokens)

            span_length = boundary - orig_index
            num_placeholders = len(markup_group)
            if num_placeholders == 1:
                allocations = [span_length]
            else:
                allocations = [1] * num_placeholders
                remaining = span_length - num_placeholders
                while remaining > 0:
                    for index in range(num_placeholders - 1, -1, -1):
                        if remaining <= 0:
                            break
                        allocations[index] += 1
                        remaining -= 1

            for group_index, placeholder in enumerate(markup_group):
                markup_content = placeholder_mapping[placeholder][0]
                tag = group_tags[group_index]
                for count in range(allocations[group_index]):
                    if is_pred and orig_index >= len(orig_tokens):
                        break
                    mapping.append(
                        (
                            orig_tokens[orig_index],
                            f"{markup_content}{count + 1}",
                            tag,
                        )
                    )
                    orig_index += 1

        return mapping

    def _get_mapping_tuple(self, original, pred, labels):
        ignored_chars = [",", ".", "'", '"', ":", ";", "*"]
        for char in ignored_chars:
            pred = pred.replace(char, "")
            labels = labels.replace(char, "")
            original = original.replace(char, "")

        pred = self._fix_tags(pred)
        labels = self._fix_tags(labels)
        original = self._fix_tags(original)

        return {
            "pred": pred,
            "labels": labels,
            "original": original,
            "pred_mapping": self._create_token_mapping(original, pred, is_pred=True),
            "label_mapping": self._create_token_mapping(original, labels),
        }

    def _fix_tags(self, text):
        pattern = r"<([^>]+)>([^<]+)</([^>]+)>([^\s<]+)\s+"
        for open_tag, content, end_tag, suffix in re.findall(pattern, text):
            tag = f"<{open_tag}>{content}</{end_tag}>"
            modified_tag = f"<{open_tag}>{content}{suffix}</{end_tag}>"
            text = text.replace(tag + suffix, modified_tag)
        return text

    def _get_embedding(self, text, model):
        import torch

        tokenizer, encoder = self._load_embedding_model(model)
        if model == "clip":
            inputs = tokenizer([text]).to(self._get_embedding_device())
        else:
            max_length = tokenizer.model_max_length
            if max_length > 100000:
                max_length = 512
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {key: value.to(encoder.device) for key, value in inputs.items()}
        with torch.no_grad():
            if model == "clip":
                embedding = encoder.encode_text(inputs)
            elif model == "bert":
                output = encoder(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                hidden = output.last_hidden_state * mask
                embedding = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                raise NotImplementedError(f"Not supported: {model}")
        return embedding.cpu().detach().numpy()

    def _load_embedding_model(self, model):
        if model in self._embedding_models:
            return self._embedding_models[model]
        import torch

        device = self._get_embedding_device()
        if model == "bert":
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            encoder = AutoModel.from_pretrained(self.bert_model_name)
        elif model == "clip":
            import clip

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                encoder, _ = clip.load(self.clip_model_name, device=device, jit=False)
            tokenizer = clip.tokenize
        else:
            raise NotImplementedError(f"Not supported: {model}")
        encoder = encoder.to(device).float()
        encoder.eval()
        self._embedding_models[model] = (tokenizer, encoder)
        return self._embedding_models[model]

    def _get_embedding_device(self):
        if self.embedding_device:
            return self.embedding_device
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for bert_f1 and clip_f1 evaluation.")
        return "cuda"

    def _compute_sim(self, text1, text2, model):
        if text1 == text2:
            return 1.0
        if model == "bert":
            from bert_score import score as bert_score

            kwargs = {
                "cands": [text1],
                "refs": [text2],
                "model_type": self.bert_model_name,
                "device": self._get_embedding_device(),
                "verbose": False,
            }
            if self.bert_num_layers is not None:
                kwargs["num_layers"] = self.bert_num_layers
            _, _, f_scores = bert_score(**kwargs)
            self._bert_score_loaded = True
            return float(f_scores.detach().cpu().numpy()[0])
        embedding1 = self._get_embedding(text1, model).flatten().astype(np.float64)
        embedding2 = self._get_embedding(text2, model).flatten().astype(np.float64)
        denominator = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if denominator == 0:
            return 0.0
        return float(
            np.dot(embedding1, embedding2) / denominator
        )


def evaluate_f1_score(original, pred, labels, mode="editor"):
    return HallucinationEvalTool("test", models=[]).evaluate_word_level(
        original, pred, labels, mode=mode, metric="f1"
    )
