from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.constants import CANN_SKILLS_DIR, REPO_ROOT
from orchestrator.workspace_runtime import _agent_runtime_directive, link_runtime


class WorkspaceRuntimeSkillTests(unittest.TestCase):
    def test_links_pinned_cann_catalog_and_custom_op_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            link_runtime(workspace)

            catalog = workspace / "cann-skills"
            self.assertTrue(catalog.is_symlink())
            self.assertEqual(catalog.resolve(), CANN_SKILLS_DIR.resolve())

            adapter = REPO_ROOT / "skills" / "ascendc-custom-pytorch-op"
            for runtime_path in (
                workspace / ".agents" / "skills" / "ascendc-custom-pytorch-op",
                workspace / ".claude" / "skills" / "ascendc-custom-pytorch-op",
                workspace / ".qoder" / "skills" / "ascendc-custom-pytorch-op",
            ):
                with self.subTest(runtime_path=runtime_path):
                    self.assertTrue(runtime_path.is_symlink())
                    self.assertEqual(runtime_path.resolve(), adapter.resolve())

            gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
            self.assertIn("/cann-skills\n", gitignore)

    def test_codex_and_pi_directives_advertise_custom_op_skill(self) -> None:
        self.assertIn("ascendc-custom-pytorch-op", _agent_runtime_directive("codex"))
        self.assertIn("ascendc-custom-pytorch-op", _agent_runtime_directive("pi"))


if __name__ == "__main__":
    unittest.main()
