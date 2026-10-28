"""Local LLM access: Ollama over requests, a scripted fake for tests, a sqlite3 cache, a cost ledger.

``chat(messages, max_tokens, tag, ref, json_mode=False, top_logprobs=0) -> ChatResult``
is all the extractor, the resolver, the summarizer and the answerer need.

The Ollama server on the reference machine also serves a production system
that keeps ``qwen2.5-coder:7b`` loaded with ``keep_alive = -1`` and
``num_ctx = 24576``. Every request therefore sends exactly those values (a
different ``num_ctx`` would make Ollama reload the model, and a finite
``keep_alive`` would let it be evicted), calls are made strictly one at a
time, and a failed call backs off for longer each time instead of retrying
at once. This module only ever POSTs to ``/api/chat``; it never loads,
pulls, unloads or stops anything.

Every call is recorded in a ledger with what it was for (``tag``: extract,
merge, summary, answer) and which episode or question it served (``ref``),
together with the prompt and output token counts and the latency measured
when the call was actually made. A cached replay returns the stored
measurement, so cost tables are the same on a replay as on the uncached run.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

REQUIRED_OPTIONS = {"num_ctx": 24576, "temperature": 0, "seed": 42}


@dataclass
class ChatResult:
    text: str
    latency_s: float
    cached: bool = False
    top_logprobs: list[dict[str, float]] = field(default_factory=list)  # per generated token: token -> logprob
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    pass


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434",
                 timeout: float = 600.0, backoff_s: tuple[float, ...] = (20.0, 60.0, 180.0)):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.backoff_s = backoff_s

    def payload(self, messages: list[dict], max_tokens: int, json_mode: bool, top_logprobs: int) -> dict:
        body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": -1,
                "options": {**REQUIRED_OPTIONS, "num_predict": max_tokens}}
        if json_mode:
            body["format"] = "json"
        if top_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        return body

    def chat(self, messages: list[dict], max_tokens: int = 256, json_mode: bool = False,
             top_logprobs: int = 0) -> ChatResult:
        body = self.payload(messages, max_tokens, json_mode, top_logprobs)
        last = None
        for attempt in range(len(self.backoff_s) + 1):
            if attempt:
                time.sleep(self.backoff_s[attempt - 1])  # back off; never hammer a server that serves customers
            t0 = time.perf_counter()
            try:
                r = requests.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
            except requests.RequestException as exc:
                last = f"request failed: {exc}"
                continue
            if r.status_code != 200:
                last = f"/api/chat returned {r.status_code}: {r.text[:300]}"
                continue
            out = r.json()
            lps = [{t["token"]: t["logprob"] for t in step.get("top_logprobs", [])}
                   for step in out.get("logprobs") or []]
            return ChatResult(text=out.get("message", {}).get("content", ""), latency_s=time.perf_counter() - t0,
                              top_logprobs=lps, prompt_tokens=int(out.get("prompt_eval_count") or 0),
                              completion_tokens=int(out.get("eval_count") or 0))
        raise LLMError(f"ollama: {last} (after {len(self.backoff_s)} back-offs)")


class ScriptedProvider:
    """Calls ``fn(messages) -> str | ChatResult`` and records every call. For the offline tests."""

    name = "scripted"
    model = "scripted"

    def __init__(self, fn: Callable[[list[dict]], "str | ChatResult"]):
        self.fn = fn
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], max_tokens: int = 256, json_mode: bool = False,
             top_logprobs: int = 0) -> ChatResult:
        self.calls.append({"messages": messages, "max_tokens": max_tokens, "json_mode": json_mode,
                           "top_logprobs": top_logprobs})
        out = self.fn(messages)
        if isinstance(out, ChatResult):
            return out
        return ChatResult(text=out, latency_s=0.01, prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
                          completion_tokens=max(1, len(out) // 4))


class LLM:
    """The one object every stage calls: optional sqlite3 cache plus the cost ledger."""

    def __init__(self, inner, cache_path: str | Path | None = None):
        self.inner = inner
        self.model = inner.model
        self.db = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(cache_path))
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.ledger: list[dict] = []
        self.allow_miss = True  # False: a cache miss raises (replay-only phases)
        self.wall_s = 0.0       # model time actually spent in this process

    def _key(self, messages, max_tokens, json_mode, top_logprobs) -> str:
        raw = json.dumps([self.inner.name, self.model, messages, max_tokens, json_mode, top_logprobs,
                          REQUIRED_OPTIONS], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def chat(self, messages: list[dict], max_tokens: int = 256, tag: str = "", ref: str = "",
             json_mode: bool = False, top_logprobs: int = 0) -> ChatResult:
        key = self._key(messages, max_tokens, json_mode, top_logprobs)
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone() if self.db else None
        if row:
            d = json.loads(row[0])
            res = ChatResult(d["text"], d["latency_s"], True, d.get("top_logprobs", []), d.get("prompt_tokens", 0),
                             d.get("completion_tokens", 0))
        else:
            if not self.allow_miss:
                raise LLMError(f"cache miss with allow_miss=False ({tag} {ref})")
            res = self.inner.chat(messages, max_tokens=max_tokens, json_mode=json_mode, top_logprobs=top_logprobs)
            self.wall_s += res.latency_s
            if self.db:
                self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps({
                    "text": res.text, "latency_s": res.latency_s, "top_logprobs": res.top_logprobs,
                    "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens})))
                self.db.commit()
        self.ledger.append({"tag": tag, "ref": ref, "key": key[:16], "cached": res.cached,
                            "latency_s": res.latency_s, "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens})
        return res

    def close(self) -> None:
        if self.db:
            self.db.close()


def build_llm(cfg: dict, cache_path: str | Path | None) -> LLM:
    return LLM(OllamaProvider(cfg.get("model", "qwen2.5-coder:7b"), cfg.get("base_url", "http://localhost:11434"),
                              float(cfg.get("timeout_s", 600))), cache_path)


def choice_probs(top_logprobs: list[dict[str, float]], options: list[str], text: str = "") -> dict[str, float]:
    """Distribution over single-token ``options`` read from the first generated token's top log-probabilities.

    Options absent from the top-k get zero mass; if none is present, the
    generated text decides (probability 1 for the option it starts with,
    uniform if none)."""
    mass = {o: 0.0 for o in options}
    if top_logprobs:
        for tok, lp in top_logprobs[0].items():
            t = tok.strip()
            if t in mass:
                mass[t] += math.exp(lp)
    total = sum(mass.values())
    if total > 0:
        return {k: v / total for k, v in mass.items()}
    t = text.strip()
    for o in options:
        if t.startswith(o):
            return {k: (1.0 if k == o else 0.0) for k in options}
    return {k: 1.0 / len(options) for k in options}
