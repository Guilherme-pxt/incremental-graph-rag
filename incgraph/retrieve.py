"""Retrieval: the hybrid baseline over episodes, and the graph retriever shared by both graphs.

**Hybrid (system 1).** BM25 and LSA over one broker's episodes, fused with
Reciprocal Rank Fusion; the top k episodes go to the answerer, listed oldest
first with their dates, so the model can tell a newer value from an older
one. BM25, LSA and RRF come from my hybrid-search-rag project (same author),
with LSA standing in for a neural embedder because the local Ollama refuses
``/api/embed``.

**Graph (systems 2 and 3).** Given a question and one broker's graph:
1. link: find the entities the question names (full names, aliases, a client's
   surname; a first name followed by a different capitalized surname does not
   link, so "Marina Lindqvist" does not pull in Marina Okafor);
2. expand: every fact touching a linked entity, then the facts touching their
   neighbours (2 hops), the second hop ranked by BM25 against the question;
3. search: when linking finds nothing, BM25 over every fact's text, as a net;
4. type tables: a question about "my clients" also gets one row per client and
   per building (current values, or every value with its date in the batch
   graph), because list and count questions need all of them, not the top k;
5. the batch graph adds the reports of the linked entities' communities.
Every fact line carries the episode ids it came from, so answers can cite them.
"""

from __future__ import annotations

import re

import numpy as np

from .bm25 import BM25
from .extract import episode_header
from .fusion import rrf
from .text import tokenize
from .values import norm_name


class LSA:
    """TF-IDF (sublinear tf, smooth idf, L2) projected on the top right singular vectors (hybrid-search-rag)."""

    def __init__(self, dims: int = 32):
        self.dims = dims

    def _tfidf(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                j = self.vocab.get(tok)
                if j is not None:
                    m[i, j] += 1
        m = np.where(m > 0, 1 + np.log(np.maximum(m, 1)), 0) * self.idf
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def fit(self, texts: list[str]) -> "LSA":
        df: dict[str, int] = {}
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] = df.get(tok, 0) + 1
        self.vocab = {tok: i for i, tok in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)], dtype=np.float32)
        x = self._tfidf(texts)
        k = max(1, min(self.dims, min(x.shape) - 1))
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        vt = vt[:k]
        signs = np.sign(vt[np.arange(k), np.abs(vt).argmax(axis=1)])
        self.proj = (vt * signs[:, None]).T.astype(np.float32)
        v = x @ self.proj
        self.vectors = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        return self

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        q = self._tfidf([query]) @ self.proj
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        s = self.vectors @ q[0]
        order = np.lexsort((np.arange(len(s)), -s))[:top_n]
        return [(int(i), float(s[i])) for i in order]


class HybridIndex:
    def __init__(self, episodes: list[dict], broker_names: dict[str, str], dims: int = 32, candidates: int = 30,
                 rrf_k: int = 60):
        self.episodes = episodes
        self.texts = [f"{episode_header(e, broker_names[e['broker']])}: {e['text']}" for e in episodes]
        self.bm25 = BM25().fit(self.texts)
        self.lsa = LSA(dims).fit(self.texts)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def search(self, question: str, k: int) -> list[int]:
        a = [i for i, _ in self.bm25.search(question, self.candidates)]
        b = [i for i, _ in self.lsa.search(question, self.candidates)]
        return [i for i, _ in rrf([a, b], self.rrf_k)][:k]

    def context(self, question: str, k: int) -> tuple[str, list[str]]:
        idx = sorted(self.search(question, k), key=lambda i: (self.episodes[i]["ts"], self.episodes[i]["id"]))
        lines = [self.texts[i] for i in idx]
        return "Messages and notes, oldest first:\n" + "\n".join(lines), [self.episodes[i]["id"] for i in idx]


# ------------------------------------------------------------------ graph retrieval

_CAP_WORD = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-zA-Z]+)")


def link(graph, question: str) -> list[str]:
    """Nodes the question names; for clients the strongest form of match wins (full name > surname > first name)."""
    qn = f" {norm_name(question)} "
    pairs = {(a.lower(), b.lower()) for a, b in _CAP_WORD.findall(question)}
    linked, person_scores = [], {}
    for nid, node in graph.nodes.items():
        names = [norm_name(n) for n in graph.names(nid)]
        if node["type"] == "client":
            score = 0
            for n in names:
                toks = n.split()
                if len(toks) >= 2 and len(toks[-1]) > 1 and f" {n} " in qn:
                    score = max(score, 3)
                elif len(toks) >= 2 and len(toks[-1]) > 1 and f" {toks[-1]} " in qn:
                    score = max(score, 2)
                elif f" {toks[0]} " in qn:
                    # a first name followed by another surname is somebody else
                    other = [b for a, b in pairs if a == toks[0] and not (len(toks) > 1 and b.startswith(toks[-1]))]
                    if not other:
                        score = max(score, 1)
            if score:
                person_scores[nid] = score
        else:
            strip = lambda s: " ".join(w for w in s.split() if w != "station")
            if any(f" {n} " in qn or (strip(n) and f" {strip(n)} " in qn) for n in names if len(n) > 1):
                linked.append(nid)
    if person_scores:
        best = max(person_scores.values())
        linked += sorted(n for n, s in person_scores.items() if s == best)
    return sorted(set(linked))


TABLE_TRIGGERS = ("my clients", "of my client", "how many", "which clients")


class GraphRetriever:
    def __init__(self, graph, reports: dict[int, str] | None = None, membership: dict[str, int] | None = None,
                 max_lines: int = 70, hop2: int = 25, search: int = 10, max_reports: int = 3):
        self.g = graph
        self.reports = reports or {}
        self.membership = membership or {}
        self.max_lines, self.hop2, self.search_k, self.max_reports = max_lines, hop2, search, max_reports
        self.texts = [graph.fact_text(e) for e in graph.edges]
        self.bm25 = BM25().fit(self.texts) if self.texts else None

    def rank(self, question: str, edge_ids: list[int], k: int) -> list[int]:
        if not edge_ids or self.bm25 is None:
            return []
        s = self.bm25.scores(question)
        return sorted(edge_ids, key=lambda i: (-float(s[i]), i))[:k]

    def context(self, question: str) -> tuple[str, list[str], dict]:
        g = self.g
        seeds = link(g, question)
        hop1 = sorted({e["id"] for n in seeds for e in g.edges_of(n)})
        neighbours = sorted({x for i in hop1 for x in (g.edges[i]["src"], g.edges[i]["dst"])
                             if x is not None and x not in seeds})
        hop2 = sorted({e["id"] for n in neighbours for e in g.edges_of(n)} - set(hop1))
        hop2 = self.rank(question, hop2, self.hop2)
        # the net is for linking failures only: once an entity is linked, facts that merely share a word or a
        # date with the question are distractors (another client's budget stated on the asked date, for one)
        found = [] if seeds else self.rank(question, list(range(len(g.edges))), self.search_k)
        sections = []
        if any(t in question.lower() for t in TABLE_TRIGGERS):
            label = "current values" if g.temporal else "every value with the date it was first stated"
            sections.append(f"All clients ({label}):\n" + "\n".join(g.summary_row(n) for n in g.of_type("client")))
            sections.append(f"All buildings ({label}):\n" + "\n".join(g.summary_row(n) for n in g.of_type("building")))
        lines, seen = [], set()
        for i in hop1 + hop2 + found:
            if i not in seen and len(lines) < self.max_lines:
                seen.add(i)
                lines.append(self.texts[i])
        if lines:
            sections.append("Facts (subject | predicate | object | validity):\n" + "\n".join(lines))
        comms = [self.membership[n] for n in seeds if n in self.membership]
        reports = [c for c in dict.fromkeys(comms) if c in self.reports][: self.max_reports]
        if reports:
            sections.append("Community reports:\n" + "\n".join(self.reports[c] for c in reports))
        cited = sorted({ep for i in seen for ep in g.edges[i]["episodes"]})
        info = {"linked": [g.nodes[n]["name"] for n in seeds], "facts": len(lines), "reports": len(reports),
                "tables": bool(sections and sections[0].startswith("All clients"))}
        return "\n\n".join(sections) if sections else "(nothing found)", cited, info
