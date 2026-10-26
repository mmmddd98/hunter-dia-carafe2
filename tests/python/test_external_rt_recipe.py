"""Validate portable experiment commands without importing training libraries."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_external_rt_recipe.py"
spec = importlib.util.spec_from_file_location("external_rt_recipe", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ExternalRTRecipeTests(unittest.TestCase):
    def arguments(self, config, recipe, target, scope=None):
        return argparse.Namespace(
            config=config,
            recipe=recipe,
            target=target,
            scope=scope,
            seed=2024,
            in_dir=Path("inputs/train"),
            validation_file=Path("inputs/validation.tsv"),
            out_dir=Path("outputs/run"),
            device="cpu",
            rt_model=None,
        )

    def test_all_recorded_recipes_use_native_trainer_options(self):
        tree = ast.parse(launcher.TRAINER.read_text(encoding="utf-8-sig"))
        native_flags = {
            argument.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        }
        for path in sorted((ROOT / "configs/external-rt").glob("*.json")):
            config = json.loads(path.read_text())
            for recipe in config["recipes"]:
                target = config.get("recipe_targets", {}).get(
                    recipe, config["targets"][0]
                )
                scopes = config.get("target_scopes", {}).get(target, [None])
                args = self.arguments(path, recipe, target, scopes[0])
                with self.subTest(config=path.name, recipe=recipe):
                    command = launcher.build_command(args, config)
                    flags = {value for value in command if value.startswith("--")}
                    self.assertFalse(flags - native_flags, flags - native_flags)
                    self.assertIn("--rt_validation_file", flags)
                    self.assertIn("--use_best_model", flags)

    def test_mismatched_recipe_target_is_rejected(self):
        path = ROOT / "configs/external-rt/heavy-finalists.json"
        config = json.loads(path.read_text())
        args = self.arguments(path, "anchor_irt/W7", "chronologer_hi")
        with self.assertRaisesRegex(ValueError, "requires target anchor_irt"):
            launcher.build_command(args, config)

    def test_invalid_config_produces_cli_error_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid.json"
            config.write_text("[]")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--config",
                    str(config),
                    "--recipe",
                    "example",
                    "--target",
                    "anchor_irt",
                    "--in-dir",
                    str(root / "inputs"),
                    "--validation-file",
                    str(root / "validation.tsv"),
                    "--out-dir",
                    str(root / "output"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must be a JSON object", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
