from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "slurm" / "train_voxcpm_malayalam_fsdp_a100_40_alex.sbatch"


class SlurmFSDPWrapperTests(unittest.TestCase):
    def test_resolves_launcher_after_slurm_spooling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            fake_repo = tmp_path / "persistent-repo"
            fake_launcher = fake_repo / "scripts" / "slurm" / "train_voxcpm_malayalam_alex.sbatch"
            fake_launcher.parent.mkdir(parents=True)

            capture = tmp_path / "launcher-argument.txt"
            fake_launcher.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                'printf "%s\\n" "$1" > "$WRAPPER_TEST_CAPTURE"\n',
                encoding="utf-8",
            )
            # Match the real generic launcher: `sbatch` only requires it to be
            # readable, so the FSDP wrapper must not assume it is executable.
            fake_launcher.chmod(0o644)

            job_env = tmp_path / "alex-sft-fsdp-smoke.env"
            job_env.write_text(f"VOXCPM_REPO_DIR={shlex.quote(str(fake_repo))}\n", encoding="utf-8")

            # Slurm executes a private copy rather than the submitted repository file.
            spooled_wrapper = tmp_path / "slurmd_spool" / "slurm_script"
            spooled_wrapper.parent.mkdir()
            shutil.copy2(WRAPPER, spooled_wrapper)
            spooled_wrapper.chmod(0o755)

            env = os.environ.copy()
            env["SLURM_SUBMIT_DIR"] = str(tmp_path)
            env["WRAPPER_TEST_CAPTURE"] = str(capture)
            subprocess.run([str(spooled_wrapper), str(job_env)], check=True, env=env)

            self.assertEqual(capture.read_text(encoding="utf-8").strip(), str(job_env))


if __name__ == "__main__":
    unittest.main()
