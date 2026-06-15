from __future__ import annotations

import subprocess
import sys

from src.rag_pipeline import RAGPipeline


def test_rag_query_returns_sources():
    pipeline = RAGPipeline.load()
    result = pipeline.query("What is the notice period in the NDA with Vendor X?")
    assert "answer" in result
    assert isinstance(result["sources"], list)
    assert result["sources"]
    assert result["confidence"] >= 0.0


def test_rag_refuses_out_of_scope_question():
    pipeline = RAGPipeline.load()
    result = pipeline.query("What is the termination fee for Vendor Z's ad hoc pilot?")
    assert "grounded evidence" in result["answer"].lower() or result["confidence"] < 0.45


def test_rag_eval_script_runs():
    out = subprocess.run(
        [sys.executable, "-m", "src.rag_pipeline", "--eval"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "precision@3=" in out.stdout

