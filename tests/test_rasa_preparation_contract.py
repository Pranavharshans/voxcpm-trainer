from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "scripts" / "prepare_rasa_malayalam.py"
SLURM_PREP = ROOT / "scripts" / "slurm" / "prepare_voxcpm_malayalam_alex.sbatch"
CONFIG = ROOT / "conf" / "voxcpm_v2" / "voxcpm_finetune_malayalam_rasa_sft_2epoch.yaml"
LORA_CONFIG = ROOT / "conf" / "voxcpm_v2" / "voxcpm_finetune_malayalam_rasa_lora_2epoch.yaml"


def _constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                result[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


class RasaPreparationContractTests(unittest.TestCase):
    def test_pins_malayalam_config_and_dataset_revision(self):
        constants = _constants(PREP)
        self.assertEqual(constants["DATASET_ID"], "ai4bharat/Rasa")
        self.assertEqual(constants["DATASET_CONFIG"], "Malayalam")
        self.assertEqual(constants["DATASET_REVISION"], "632f55c7ac590219d41cd7adffce5b440e4604f5")
        self.assertEqual(constants["TRAIN_SPLIT"], "train")
        self.assertEqual(constants["VALIDATION_SPLIT"], "test")

    def test_slurm_profile_uses_rasa_manifest_and_config(self):
        content = SLURM_PREP.read_text(encoding="utf-8")
        self.assertIn("rasa-sft)", content)
        self.assertIn("rasa-lora)", content)
        self.assertIn("datasets/rasa-malayalam", content)
        self.assertIn("voxcpm_finetune_malayalam_rasa_sft_2epoch.yaml", content)
        self.assertIn("voxcpm_finetune_malayalam_rasa_lora_2epoch.yaml", content)
        self.assertIn('HF_TOKEN=$(<"${HF_TOKEN_FILE}")', content)

    def test_training_profile_is_full_sft_with_native_prompt(self):
        content = CONFIG.read_text(encoding="utf-8")
        self.assertIn("num_epochs: 2", content)
        self.assertIn("distributed_strategy: fsdp", content)
        self.assertIn("eval_prompt_audio: ${MALAYALAM_EVAL_PROMPT_AUDIO}", content)
        self.assertIn("eval_prompt_text: ${MALAYALAM_EVAL_PROMPT_TEXT}", content)
        self.assertNotIn("\nlora:", content)

    def test_lora_profile_is_a_two_epoch_single_gpu_control(self):
        content = LORA_CONFIG.read_text(encoding="utf-8")
        self.assertIn("num_epochs: 2", content)
        self.assertIn("global_batch_size: 16", content)
        self.assertIn("max_batch_tokens: 6144", content)
        self.assertIn("eval_prompt_audio: ${MALAYALAM_EVAL_PROMPT_AUDIO}", content)
        self.assertIn("eval_prompt_text: ${MALAYALAM_EVAL_PROMPT_TEXT}", content)
        self.assertIn("\nlora:", content)
        self.assertIn("  r: 32", content)
        self.assertIn("  alpha: 32", content)
        self.assertNotIn("distributed_strategy: fsdp", content)


if __name__ == "__main__":
    unittest.main()
