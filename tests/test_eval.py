import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from trainers.eval import EvalTarget, HallucinationEvalTool, evaluate_f1_score


class EvaluateF1ScoreTest(unittest.TestCase):
    def make_image(self, directory):
        path = Path(directory) / "image.jpg"
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 28, 54, 44), fill="red")
        image.save(path)
        return str(path)

    def require_cuda(self):
        import torch

        self.assertTrue(torch.cuda.is_available(), "CUDA is required for this test")

    def assert_scores(self, actual, expected):
        self.assertAlmostEqual(actual[0], expected[0], places=6)
        self.assertAlmostEqual(actual[1], expected[1], places=6)
        self.assertAlmostEqual(actual[2], expected[2], places=6)

    def test_original_eval_cases(self):
        cases = [
            ("A B C", "A B C", "A B C", (0.0, 0.0, 0.0)),
            (
                "This is ABC Center in front of an apple",
                "This is <named_entities_fact>Tokyo Tower</named_entities_fact> in front of an apple",
                "This is <named_entities_fact>Tokyo Tower</named_entities_fact> in front of an apple",
                (100.0, 100.0, 100.0),
            ),
            (
                "This is apple",
                "This is <named_entities_fact>apple</named_entities_fact>",
                "This is apple",
                (0.0, 0.0, 0.0),
            ),
            (
                "This is apple",
                "This is <named_entities_fact>apple</named_entities_fact>.",
                "This is <named_entities_fact>apple</named_entities_fact>,",
                (100.0, 100.0, 100.0),
            ),
            (
                "A B C D",
                "A <tag>X</tag><tag>Y</tag> D",
                "A <tag>X</tag><tag>Y</tag> D",
                (100.0, 100.0, 100.0),
            ),
            (
                "A B C D",
                "A <tag>X</tag><tag>Y</tag> D",
                "A <tag>Z</tag><tag>Y</tag> D",
                (50.0, 50.0, 50.0),
            ),
            ("A B C", "A B X", "A B C", (0.0, 0.0, 0.0)),
            (
                "I love New York City in the summer",
                "I love <location>Los Angeles</location> in the <season>winter</season>",
                "I love <location>Los Angeles</location> in the <season>winter</season>",
                (100.0, 100.0, 100.0),
            ),
            (
                "This is ABC Center in front of an apple",
                "This is <named_entities_fact>Tokyo Tower</named_entities_fact> in front of an apple",
                "This is <named_entities_fact>Tokyo Tower</named_entities_fact> in front of an apple",
                (100.0, 100.0, 100.0),
            ),
        ]

        for original, pred, labels, expected in cases:
            with self.subTest(pred=pred, labels=labels):
                self.assert_scores(evaluate_f1_score(original, pred, labels), expected)

    def test_token_count_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            evaluate_f1_score("A B", "A B", "A", mode="editor")

    def test_additional_group_cases(self):
        cases = [
            ("A B C", "A B C", "A B C", (0.0, 0.0, 0.0)),
            ("A B C", "A <tag>X</tag> C", "A <tag>X</tag> C", (100.0, 100.0, 100.0)),
            ("A B C", "A <tag>X</tag> C", "A <tag>XWrong</tag> C", (0.0, 0.0, 0.0)),
            (
                "A B C D E",
                "A <tag>X</tag> C <tag>Y</tag> E",
                "A <tag>X</tag> C <tag>Z</tag> E",
                (50.0, 50.0, 50.0),
            ),
            (
                "A B C D E",
                "A <tag>X</tag> C <tag>Y</tag> E",
                "A <tag>W</tag> C <tag>Z</tag> E",
                (0.0, 0.0, 0.0),
            ),
            (
                "A B C D E F G",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G",
                "A <tag>X</tag> C <tag>Wrong</tag> E <tag>Z</tag> G",
                (200.0 / 3, 200.0 / 3, 200.0 / 3),
            ),
            (
                "A B C D E F G",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G",
                "A <tag>Wrong</tag> C <tag>Y</tag> E <tag>Wrong</tag> G",
                (100.0 / 3, 100.0 / 3, 100.0 / 3),
            ),
            (
                "A B C D E F G H I",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G <tag>W</tag> I",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G <tag>Wrong</tag> I",
                (75.0, 75.0, 75.0),
            ),
            (
                "A B C D E F G H I",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G <tag>W</tag> I",
                "A <tag>Wrong</tag> C <tag>Y</tag> E <tag>Wrong</tag> G <tag>W</tag> I",
                (50.0, 50.0, 50.0),
            ),
            (
                "A B C D E F G H I J K",
                "A <tag>X</tag> C <tag>Y</tag> E <tag>Z</tag> G <tag>W</tag> I <tag>V</tag> K",
                "A <tag>X</tag> C <tag>Wrong</tag> E <tag>Z</tag> G <tag>W</tag> I <tag>V</tag> K",
                (80.0, 80.0, 80.0),
            ),
        ]

        for original, pred, labels, expected in cases:
            with self.subTest(pred=pred, labels=labels):
                self.assert_scores(evaluate_f1_score(original, pred, labels), expected)

    def test_editor_evaluation_cases(self):
        evaltool = HallucinationEvalTool("test", models=[])
        cases = [
            (
                "A red train is at the station",
                "A <object>blue bus</object> is at the station",
                "A <object>blue bus</object> is at the station",
                (100.0, 100.0, 100.0),
            ),
            (
                "A red train beside a green car",
                "A <color>blue</color> train beside a <object>truck</object>",
                "A <color>blue</color> train beside a <object>bus</object>",
                (50.0, 50.0, 50.0),
            ),
            (
                "A red train is at the station",
                "A <object>blue bus</object> is at the station",
                "A <object>green truck</object> is at the station",
                (0.0, 0.0, 0.0),
            ),
            (
                "A red train is at the station",
                "A <object>blue bus</object> is at the station",
                "A red train is at the station",
                (0.0, 0.0, 0.0),
            ),
        ]

        for original, pred, labels, expected in cases:
            with self.subTest(pred=pred, labels=labels):
                actual = evaltool.evaluate_word_level(
                    original, pred, labels, mode="editor", metric="f1"
                )
                self.assert_scores(actual, expected)

    def test_bert_and_clip_f1_load_real_models(self):
        self.require_cuda()
        original = "A red train beside a green car"
        pred = "A <object>blue bus</object> beside a <color>yellow</color> car"
        labels = "A <object>green truck</object> beside a <color>yellow</color> car"

        evaltool = HallucinationEvalTool(
            "test",
            models=[],
            embedding_device="cuda",
        )
        self.assertEqual(evaltool.bert_model_name, "microsoft/deberta-xlarge-mnli")
        self.assertEqual(evaltool.bert_num_layers, 40)
        self.assertEqual(evaltool.clip_model_name, "ViT-B/32")
        for metric in ["bert", "clip"]:
            actual = evaltool.evaluate_word_level(
                original, pred, labels, mode="editor", metric=metric
            )
            for score in actual:
                self.assertTrue(math.isfinite(score))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 100.000001)
            if metric == "bert":
                self.assertTrue(evaltool._bert_score_loaded)
            else:
                self.assertIn(metric, evaltool._embedding_models)
                _, encoder = evaltool._embedding_models[metric]
                self.assertEqual(next(encoder.parameters()).device.type, "cuda")

    def test_run_reports_bert_and_clip_f1_scores(self):
        self.require_cuda()
        evaltool = HallucinationEvalTool(
            "test",
            models=[],
            embedding_device="cuda",
        )
        targets = [
            EvalTarget(
                pred="A blue bus beside a yellow car",
                cand="A red train beside a green car",
                refs=["A green truck beside a yellow car"],
                image_path="image-a.jpg",
                pred_with_tags="A <object>blue bus</object> beside a <color>yellow</color> car",
                labels_with_tags="A <object>green truck</object> beside a <color>yellow</color> car",
            )
        ]

        mean_scores, all_scores = evaltool.run(
            targets, mode="editor", metrics=["bert-f1", "clip-f1"]
        )

        for metric_name in ["bert_f1", "clip_f1"]:
            self.assertIn(metric_name, mean_scores)
            self.assertIn(metric_name, all_scores)
            self.assertEqual(len(all_scores[metric_name]), 1)
            self.assertTrue(math.isfinite(mean_scores[metric_name]))
            self.assertTrue(math.isfinite(all_scores[metric_name][0]))

    def test_sentence_level_uses_capevalkit(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = self.make_image(directory)
            evaltool = HallucinationEvalTool(
                "test",
                models=[],
                capevalkit_output_dir=Path(directory) / "capevalkit",
            )
            targets = [
                EvalTarget(
                    pred="a red rectangle",
                    cand="a red rectangle",
                    refs=["a red rectangle"],
                    image_path=image_path,
                    pred_with_tags="a red rectangle",
                    labels_with_tags="a red rectangle",
                )
            ]
            mean_scores, all_scores = evaltool.evaluate_sentence_level(
                targets, metrics=["pacscore"]
            )

            self.assertIn("PAC-S", mean_scores)
            self.assertIn("PAC-S", all_scores)
            self.assertEqual(len(all_scores["PAC-S"]), 1)
            self.assertTrue(math.isfinite(mean_scores["PAC-S"]))
            self.assertTrue(math.isfinite(all_scores["PAC-S"][0]))

    def test_run_combines_word_and_capevalkit_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = self.make_image(directory)
            evaltool = HallucinationEvalTool(
                "test",
                models=[],
                capevalkit_output_dir=Path(directory) / "capevalkit",
            )
            targets = [
                EvalTarget(
                    pred="a red rectangle",
                    cand="A red car is here",
                    refs=["a red rectangle"],
                    image_path=image_path,
                    pred_with_tags="A <object>blue bus</object> is here",
                    labels_with_tags="A <object>blue bus</object> is here",
                ),
                EvalTarget(
                    pred="a red rectangle",
                    cand="A red car is here",
                    refs=["a red rectangle"],
                    image_path=image_path,
                    pred_with_tags="A <object>green train</object> is here",
                    labels_with_tags="A <object>blue bus</object> is here",
                ),
            ]
            mean_scores, all_scores = evaltool.run(
                targets, mode="editor", metrics=["f1", "pacscore"]
            )

            self.assertEqual(mean_scores["f1_f1"], 50.0)
            self.assertEqual(all_scores["f1_f1"], [100.0, 0.0])
            self.assertIn("PAC-S", mean_scores)
            self.assertIn("PAC-S", all_scores)
            self.assertEqual(len(all_scores["PAC-S"]), 2)
            self.assertTrue(math.isfinite(mean_scores["PAC-S"]))


if __name__ == "__main__":
    unittest.main()
