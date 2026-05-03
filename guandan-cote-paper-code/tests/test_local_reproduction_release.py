# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_release_text() -> str:
    chunks: list[str] = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if path.parts[-2:] and any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".json"}:
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


class LocalReproductionReleaseTest(unittest.TestCase):
    def test_release_surface_has_no_removed_api_backend(self) -> None:
        text = read_release_text()
        removed_name = "deep" + "seek"
        self.assertIsNone(re.search(removed_name, text, flags=re.IGNORECASE))
        self.assertNotIn("DEEP" + "SEEK_", text)

    def test_release_surface_has_no_hard_coded_paper_result_constants(self) -> None:
        text = read_release_text()
        self.assertNotIn("PAPER_" + "COTE_WIN_RATE", text)
        self.assertNotIn("PAPER_" + "EDGE_COUNT", text)
        self.assertNotIn("PAPER_" + "EDGE_RETENTION", text)
        self.assertNotIn("relative_to_" + "paper_cote", text)
        self.assertNotIn("paper_" + "cote_win_rate", text)

    def test_local_training_modules_import_without_heavy_runtime_dependencies(self) -> None:
        from cote_paper.local_model import LocalCausalLM
        from cote_paper.soft_prompt import SoftPromptTrainer

        backend = LocalCausalLM(model_path="")
        trainer = SoftPromptTrainer()
        self.assertFalse(backend.enabled)
        self.assertEqual(trainer.last_update_count, 0)

    def test_local_edge_model_defaults_match_reproduction_path(self) -> None:
        from cote_paper.config import HyperParams

        hp = HyperParams()
        self.assertTrue(hp.edge_local_model)
        self.assertEqual(hp.edge_local_model_budget, 56)

    def test_topology_runtime_summary_reports_current_state_only(self) -> None:
        from cote_paper.topology import CoteTopologyState

        summary = CoteTopologyState.dense_initial().runtime_summary()
        self.assertIn("current_edge_count", summary)
        self.assertIn("current_edge_retention", summary)
        self.assertNotIn("paper_" + "edge_count", summary)
        self.assertNotIn("paper_" + "edge_retention", summary)


if __name__ == "__main__":
    unittest.main()
