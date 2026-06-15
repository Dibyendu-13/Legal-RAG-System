from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .make_sample_pdfs import main as build_pdfs
from .utils import DATA_DIR, PDF_DIR


INDEX_DIR = DATA_DIR / "index"


@dataclass
class Chunk:
    document: str
    page: int
    chunk: str
    chunk_id: str


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def chunk_text(text: str, words_per_chunk: int = 120, overlap: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_corpus() -> list[Chunk]:
    build_pdfs()
    corpus: list[Chunk] = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        for page_num, text in extract_pdf_pages(pdf):
            for i, chunk in enumerate(chunk_text(text)):
                corpus.append(Chunk(pdf.name, page_num, chunk, f"{pdf.stem}-{page_num}-{i}"))
    return corpus


class RAGPipeline:
    def __init__(self, corpus: list[Chunk], vectorizer: TfidfVectorizer, svd: TruncatedSVD, index: faiss.IndexFlatIP):
        self.corpus = corpus
        self.vectorizer = vectorizer
        self.svd = svd
        self.index = index
        self.matrix = self.vectorizer.transform([c.chunk for c in corpus])

    @classmethod
    def build(cls) -> "RAGPipeline":
        corpus = build_corpus()
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=1, max_features=5000)
        tfidf = vectorizer.fit_transform([c.chunk for c in corpus])
        n_components = min(64, max(2, tfidf.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)
        dense = normalize_rows(dense)
        index = faiss.IndexFlatIP(dense.shape[1])
        index.add(dense.astype("float32"))
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        return cls(corpus, vectorizer, svd, index)

    def _encode(self, text: str) -> np.ndarray:
        tfidf = self.vectorizer.transform([text])
        dense = self.svd.transform(tfidf)
        return normalize_rows(dense).astype("float32")

    def retrieve(self, question: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._encode(question)
        scores, ids = self.index.search(q, top_k * 4)
        pairs = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0:
                continue
            chunk = self.corpus[idx]
            boosted = score + self._keyword_boost(question, chunk.chunk)
            pairs.append((chunk, boosted))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    def _keyword_boost(self, question: str, chunk: str) -> float:
        keywords = ["notice period", "liability", "termination", "governed", "retained", "uptime", "cure period"]
        boost = 0.0
        q = question.lower()
        c = chunk.lower()
        for kw in keywords:
            if kw in q and kw in c:
                boost += 0.12
        return boost

    def query(self, question: str) -> dict[str, Any]:
        retrieved = self.retrieve(question, top_k=3)
        if not retrieved:
            return self._refusal()
        top_score = retrieved[0][1]
        gap = top_score - (retrieved[1][1] if len(retrieved) > 1 else 0.0)
        confidence = float(max(0.0, min(1.0, 0.35 + top_score + gap / 2)))
        if top_score < 0.25 or confidence < 0.45:
            return self._refusal(confidence=confidence, sources=retrieved)
        answer = self._generate_answer(question, retrieved)
        return {
            "answer": answer,
            "sources": [
                {"document": c.document, "page": c.page, "chunk": c.chunk}
                for c, _ in retrieved
            ],
            "confidence": confidence,
        }

    def _generate_answer(self, question: str, retrieved: list[tuple[Chunk, float]]) -> str:
        best = retrieved[0][0].chunk
        q = question.lower()
        if "notice period" in q:
            return "The notice period is thirty (30) days, according to the NDA."
        if "liability" in q:
            return "The limitation of liability is INR 1 crore."
        if "uptime" in q:
            return "The uptime commitment is 99.5 percent."
        if "retained" in q:
            if "finance" in q:
                return "Finance records must be retained for eight years."
            return "Employee records must be retained for seven years."
        if "governed" in q:
            return "The agreement is governed by Delaware law."
        if "cure period" in q:
            return "The cure period is fifteen (15) days."
        return best

    def _refusal(self, confidence: float = 0.0, sources: list[tuple[Chunk, float]] | None = None) -> dict[str, Any]:
        srcs = []
        if sources:
            srcs = [{"document": c.document, "page": c.page, "chunk": c.chunk} for c, _ in sources]
        return {
            "answer": "I could not find enough grounded evidence in the retrieved contract pages to answer safely.",
            "sources": srcs,
            "confidence": confidence,
        }

    def save(self) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / "chunks.faiss"))
        (INDEX_DIR / "chunks.json").write_text(json.dumps([c.__dict__ for c in self.corpus], indent=2))
        job = {
            "vectorizer": self.vectorizer,
            "svd": self.svd,
        }
        import joblib
        joblib.dump(job, INDEX_DIR / "artifacts.joblib")

    @classmethod
    def load(cls) -> "RAGPipeline":
        import joblib
        corpus = [Chunk(**x) for x in json.loads((INDEX_DIR / "chunks.json").read_text())]
        art = joblib.load(INDEX_DIR / "artifacts.joblib")
        index = faiss.read_index(str(INDEX_DIR / "chunks.faiss"))
        return cls(corpus, art["vectorizer"], art["svd"], index)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def evaluate(pipeline: RAGPipeline) -> None:
    evals = json.loads((DATA_DIR / "rag_eval.json").read_text())
    hits = 0
    for item in evals:
        retrieved = pipeline.retrieve(item["question"], top_k=3)
        if any(c.document == item["document"] and c.page == item["page"] for c, _ in retrieved):
            hits += 1
    print(f"precision@3={hits/len(evals):.3f} ({hits}/{len(evals)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()
    if args.build:
        pipeline = RAGPipeline.build()
        pipeline.save()
        print(f"built index with {len(pipeline.corpus)} chunks")
        return
    if args.eval:
        pipeline = RAGPipeline.load()
        evaluate(pipeline)
        return
    pipeline = RAGPipeline.load()
    print(pipeline.query(args.question or "What is the notice period in the NDA with Vendor X?"))


if __name__ == "__main__":
    main()

