# An incrementally growing knowledge graph against batch GraphRAG and hybrid search, measured

A real-estate broker's knowledge is a stream, not a corpus. Every WhatsApp message and every note adds something, and much of it replaces something older: a client raises the budget, drops one building for another, needs a third bedroom; a developer changes the starting price or delays the keys. Retrieval over the raw messages returns the old budget next to the new one. A GraphRAG index built in batch knows who is who, but it is a function of the whole corpus: every new message means building it again, and it has no notion of a fact that stopped being true. This project tests a third design, the one behind the idea of "an Obsidian-style knowledge graph for each broker, growing with every conversation": each new message is read once, its entities are matched to the nodes the graph already has, and a fact that changed gets an end date instead of being overwritten. The question is whether that graph answers better than plain hybrid search, at a per-message cost far below rebuilding, and where it loses.

Everything runs locally: Python standard library plus numpy and requests, and a local Ollama server with one 7B model. The brokerage, its brokers, clients, buildings and city are synthetic and fictional (**Velmora**, an invented city).

**Result in one paragraph.** On 81 gold questions over a 130-episode stream, the incremental temporal graph answered 74.1% correctly against 61.7% for hybrid search (+12.3 points, paired bootstrap 95% CI [+2.5, +22.2]) and 60.5% for batch GraphRAG (+13.6 [+4.9, +22.2]). The gain comes from questions over many entities (aggregation: 66.7% against 0.0% for hybrid) and from entity resolution: the graph ended with 0 wrong merges, 2.6% duplicate nodes against 36.4% for the batch graph's name-only merging, and it closed all 33 superseded facts on the exact date. It did **not** win on "now" questions: 86.7% against 93.3% for hybrid, and its stale-fact rate on questions whose answer had changed was 33.3% (7 of 21) against 4.8% for hybrid. In every case checked the graph held the right current value; the 7B answerer picked the superseded line listed above it. Multi-hop questions were bad for all three systems (0 to 2 correct out of 12). Isolation held: 0 leaks out of 9 probes for every per-broker system, against 8 leaks out of 9 for one shared index. Absorbing the last 32 episodes cost the incremental graph 34 LLM calls, 21,313 tokens and 71 s of model time (2.2 s per new episode); rebuilding the batch graph cost 146 calls, 101,826 tokens and 1,106 s, or 389 s even with every extraction replayed from a cache; re-indexing hybrid search cost no LLM call at all. Full uncached run: 43 minutes on an Apple M5 laptop with 16 GB of RAM, almost all of it model time on a server shared with a production system.

## Architecture

```mermaid
flowchart LR
    S[(episode stream<br/>chats and notes,<br/>one broker)] --> X[extract<br/>1 LLM call per episode<br/>subject / predicate / object lines]
    X --> R{resolve each name}
    R -- chat thread key --> N[node]
    R -- exact name or alias --> N
    R -- rules: nicknames,<br/>surname initial,<br/>building word prefix --> N
    R -- still ambiguous --> L[LLM picks a candidate<br/>or 'new', 1 token] --> N
    R -- no candidate --> NEW[new node] --> N
    N --> T[assert fact<br/>single-valued and changed:<br/>close old edge valid_to,<br/>open new edge valid_from]
    T --> G[(temporal graph<br/>one per broker,<br/>plain dicts)]
    Q[question] --> K[link entities<br/>names, aliases, surnames]
    G --> K
    K --> H[2-hop expansion,<br/>BM25 over edge facts,<br/>client and building tables]
    H --> A[answer, qwen2.5-coder:7b<br/>ANSWER: ... with episode ids]
```

The two comparison systems read the same stream. **Batch GraphRAG** makes the same extraction call per episode, then merges entities by normalized name, keeps every fact with the dates it was stated, finds communities (Louvain) and writes one LLM report per community; it is rebuilt when new episodes arrive. **Hybrid search** indexes the raw episodes with BM25 and LSA fused by Reciprocal Rank Fusion. All three answer with the same prompt and the same model; only the context block differs.

## Design decisions, and why

**A stream built around the failure modes** (`incgraph/synth.py`, `incgraph/questions.py`). Three brokers, seven clients each, eight buildings in a shared catalogue (developer, neighborhood, metro station, bedroom counts, starting price, delivery month). Each broker's stream runs from January to June 2026: listing notes, client chats (the client writes in the first person; the chat header names the contact as saved in the broker's phone), the broker's own notes, a few replies and office notes with no facts. 130 episodes, 293 true facts. Facts change on purpose: every broker has a client who raises the budget twice, one who switches buildings, one who needs another bedroom, one who moves the deadline, one who cuts the budget and switches, two buildings whose price changes and one whose delivery slips (33 superseded facts in all). Confounders on purpose: Marina Okafor (broker A) and Marina Lindqvist (broker B); Daniel Reyes and Daniel Weber at the same broker; chat contacts saved as "Kate Moreau", "Fil Arantes", "Dani Reyes", "Jonah P" for clients whose own messages and the broker's notes give the full names Katherine Moreau, Felipe Arantes, Daniel Reyes and Jonah Pellegrini; buildings written as "N1", "the Lofts", "Aurora Res.", "the Gardens", and two buildings sharing a word, so "the Aurora" is ambiguous without context. The generator keeps the ground truth: every entity with every surface form used for it, every fact with its validity interval and the episode that stated it, and per episode which entity each surface form refers to. Output is byte-identical for a seed.

The 81 gold questions (27 per broker) are typed: 15 single-hop (developer, station, wanted neighborhood, buildings visited, an objection), 15 temporal "now" (budget, building of interest, bedrooms, price, delivery after they changed), 12 "as of" a past date, 12 multi-hop (the developer of the building a client is focused on now; which clients are interested in a building near a station; the owner's example, "which of my clients looking for 3 bedrooms can afford a building delivered in 2027 near Riverside station?"), 9 aggregation (counts and the highest budget, over current values), 9 unanswerable near misses (a building that does not exist, a real building this broker never listed, an attribute nobody stated) and 9 isolation probes (broker A asked for the budget of broker B's client, the Marina pair first). Gold answers are atoms derived from the ground truth (an amount, a month, a count, entities, a keyword), plus the values the same slot held at other times (for the stale-fact rate) and, for isolation probes, the other broker's values (for leaks).

**One extraction call per episode, written as lines** (`incgraph/extract.py`). A fixed schema of 13 predicates over 5 entity types, four short examples with invented names, up to two earlier messages of the same chat as context, and one `subject | predicate | object` line per fact. The first version asked for JSON and the model pretty-printed it: about 200 output tokens for a listing note, at the 8.7 tokens per second this shared 7B server produced, 20 to 40 s per episode. The same facts as lines cost about 40 to 100 tokens. Both graph systems use exactly this call, so they start from identical extractions and the comparison isolates what happens after extraction: entity resolution, time, communities, and update cost.

**Entity resolution: deterministic first, the LLM only when ambiguous** (`incgraph/resolve.py`). In order: (1) the chat thread, because a WhatsApp thread is a phone number, the strongest key a broker has; (2) exact normalized name or alias of the same type; (3) rules: compatible first names (equal, a small table of common English nicknames, or a 3-letter prefix) with the same surname or a surname initial, building names whose every word starts a word of the candidate, places by their words; (4) the LLM, when the rules leave more than one candidate or only a weak one (a bare first name, the same surname with a first name the table does not know): it sees the message, the chat context and what the graph knows about each candidate, and answers with one digit, read from the log-probabilities; 0 means "new". A surface form becomes an alias only if it pointed to one candidate, so "the Aurora" is never learned as a name. The nickname table has generic English nicknames only ("Kate", "Dani", "Liz"), not the Portuguese ones in the data ("Fil", "Bia"), so those reach the LLM or the thread key.

**Facts get an end date instead of being overwritten** (`incgraph/graph.py`). Edges carry `valid_from`, `valid_to`, `invalidated_by` and every episode id that stated them. A new value for a single-valued predicate (budget, bedrooms, wanted neighborhood, deadline, building of interest, developer, location, station, price, delivery) closes the open edge at the new episode's date; multi-valued ones (visits, objections, bedroom counts a building offers) accumulate. Restating a value adds the episode to the edge and changes nothing else. The graph is a few dicts; there is one per broker and no structure is shared, so a question to one broker's graph cannot reach another's data. That isolation is by construction, and it is the reason the isolation probes can only fail by confusing two of the broker's own entities.

**Retrieval over the graph** (`incgraph/retrieve.py`). Link the entities the question names (full names and aliases, a client's surname; a first name followed by a different capitalized surname does not link, so "Marina Lindqvist" does not pull in Marina Okafor); take every fact touching them and the 25 facts touching their neighbours that best match the question by BM25 (2 hops); when nothing links, BM25 over all fact lines as a net. Questions about "my clients" or "how many" also get one row per client and per building, because lists and counts need all of them, not the top k: current values in the incremental graph, every value with the date it was first stated in the batch graph. Each fact line says whether it is current or superseded and when, and carries its episode ids for citation.

**Batch GraphRAG, scaled down** (`incgraph/batch.py`). The indexing recipe of Microsoft's GraphRAG (Edge et al., 2024): extraction per text unit (here, the same per-episode call), entities merged by normalized name and type as GraphRAG merges by title, every distinct fact kept with the dates it was stated (a batch index has no notion of a fact ending), Louvain communities (GraphRAG uses Leiden; at about 40 nodes per broker the difference does not matter), and one LLM report of at most 4 sentences per community of 3 or more entities, told to give the most recent value when values conflict. Retrieval is the graph retrieval above plus the reports of the linked entities' communities. GraphRAG's global map-reduce search over all reports was not built; the client and building tables stand in for it on list and count questions.

**Hybrid baseline** (`incgraph/retrieve.py`, `bm25.py`, `fusion.py`, `text.py`). BM25 (k1 1.2, b 0.75), LSA (32 dimensions) and RRF (k 60) over one broker's episodes, top 8 to the answerer, listed oldest first with dates and the chat header so the model can tell a newer value from an older one. The BM25, LSA, RRF and tokenizer code comes from my hybrid-search-rag project (same author). LSA again stands in for a neural embedder: the local Ollama refuses `/api/embed`, and nothing could be installed. A second hybrid index over all three brokers' episodes answers only the isolation probes, to show what per-broker isolation prevents.

**One answer prompt** (`incgraph/answer.py`). The broker's name, today's date (2026-06-30), five rules (use only the context; a different surname is a different person; "now" means the latest value, a past date means the value that held then; lists and counts; cite episode ids) and a one-line reply, `ANSWER: <short answer> [episode ids]`. Fixed text first, context and question last, so consecutive calls share a prompt prefix.

**Scoring is deterministic** (`incgraph/scoring.py`, `incgraph/graphq.py`). An answer line is correct when it does not decline, contains every required atom (amounts in any of "$968,000", "968k", "968 thousand"; months as "June 2027", "Jun 2027", "2027-06"; counts as digits or words; entities by any unambiguous name) and, for lists, names no client or building that should not be there; questions with no answer are correct only when declined. Stale: wrong, and states a value the slot held at another time. Leak: an isolation probe whose answer states the other broker's value for that client. Graph quality maps every node to a true entity through the per-episode ground truth of which surface form means which entity, then counts wrong merges (a node holding mentions of two true entities), duplicates (extra nodes per true entity), relation precision and recall against every true fact including history, the share of true single-valued slots whose open values are exactly the true current value, and the share of superseded facts closed with the exact date. Paired bootstrap, 10,000 resamples, seed 42, as in hybrid-search-rag.

**The shared Ollama server.** The machine's Ollama also serves a production system that keeps `qwen2.5-coder:7b` loaded with `keep_alive -1` and `num_ctx 24576`. Every request sends `"keep_alive": -1` and the options `{"num_ctx": 24576, "temperature": 0, "seed": 42}` (plus `num_predict`), because another `num_ctx` would force a reload; calls are strictly sequential; a failed call waits 20 s, 60 s, then 180 s before giving up; and the code only ever POSTs to `/api/chat` (`/api/ps` is read before and after the run and recorded in `results/run_meta.json`). Every call goes through a sqlite3 cache keyed on the full request and is logged in a ledger (`results/llm_ledger.jsonl`) with its purpose, the episode or question it served, its token counts and the latency measured when it was really made, so cost tables are identical on a replay.
## Results

All numbers come from one uncached `python3 -m incgraph run` (2,555 s wall time, 402 uncached LLM calls, 2,539 s of model time; `results/run_meta.json`): 718 s of extraction, 389 s of community reports, 5 s of merge decisions and 1,428 s of answers. `python3 -m incgraph eval` replays that run from the cache in 2 s and reproduces every metric, graph-quality number and answer byte for byte. Artifacts are under `results/`: `report.md` has every table, `answers.jsonl` every context size, answer and score, `graphs/` both graphs of every broker, `resolution_log.jsonl` every resolution decision with its route, `extractions.jsonl` every raw extraction, and `llm_ledger.jsonl` every call. Model: `qwen2.5-coder:7b` (Q4_K_M) through Ollama 0.40.10, temperature 0, seed 42.

### Answers (81 questions, 27 per broker)

| system | accuracy (95% CI) | answerable (63) | false "I don't know" | stale, "now" (21) | wrong period, "as of" (12) | unanswerable declined (9) | isolation leaks (9 probes) | correct answers citing a gold episode | median prompt tokens |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid (BM25 + LSA + RRF over episodes) | 61.7% [50.6, 71.6] | 50.8% | 6.3% | **4.8%** | 25.0% | 100% | 0 | 96.9% | 714 |
| Batch GraphRAG | 60.5% [49.4, 71.6] | 49.2% | 15.9% | 33.3% | **8.3%** | 100% | 0 | 100% | 1,260 |
| Incremental temporal graph | **74.1%** [64.2, 82.7] | **66.7%** | **4.8%** | 33.3% | 25.0% | 100% | 0 | 100% | 1,266 |
| *one hybrid index shared by all brokers (isolation probes only)* | | | | | | | **8** | | |

"Stale, now" is over the 21 questions whose answer slot changed over time (15 temporal "now" questions and 6 multi-hop "now" questions): wrong, and stating a value the slot held earlier. "Wrong period" is the same for "as of" questions. Every isolation probe was declined by all three per-broker systems.

By question type:

| system | single-hop (15) | temporal "now" (15) | as of (12) | multi-hop (12) | aggregation (9) | unanswerable (9) | isolation (9) |
|---|---|---|---|---|---|---|---|
| Hybrid | 86.7% | **93.3%** | 33.3% | 8.3% | 0.0% | 100% | 100% |
| Batch GraphRAG | 93.3% | 66.7% | 25.0% | 0.0% | 44.4% | 100% | 100% |
| Incremental temporal graph | **100%** | 86.7% | **50.0%** | **16.7%** | **66.7%** | 100% | 100% |

Paired bootstrap on accuracy, points (95% CI, 10,000 resamples over the same questions):

| comparison | all 81 | single-hop | temporal "now" | as of | multi-hop | aggregation |
|---|---|---|---|---|---|---|
| incremental vs hybrid | **+12.3 [+2.5, +22.2]** | +13.3 [0.0, +33.3] | -6.7 [-26.7, +13.3] | +16.7 [-25.0, +50.0] | +8.3 [0.0, +25.0] | **+66.7 [+33.3, +100]** |
| incremental vs batch | **+13.6 [+4.9, +22.2]** | +6.7 [0.0, +20.0] | +20.0 [-13.3, +53.3] | +25.0 [0.0, +50.0] | +16.7 [0.0, +41.7] | +22.2 [0.0, +55.6] |
| batch vs hybrid | -1.2 [-9.9, +8.6] | +6.7 [0.0, +20.0] | **-26.7 [-53.3, -6.7]** | -8.3 [-41.7, +25.0] | -8.3 [-25.0, 0.0] | **+44.4 [+11.1, +77.8]** |

### Graph quality against the ground truth (3 brokers pooled)

| graph | nodes | wrong merges | duplication rate | clients: nodes / true | buildings: nodes / true | relation precision | relation recall | current-value accuracy (195 slots) | superseded facts closed, exact date (33) |
|---|---|---|---|---|---|---|---|---|---|
| Incremental temporal graph | 80 | 0 | 2.6% | 21 / 21 | 20 / 18 | 96.6% | 98.6% | **97.9%** | **33 (33)** |
| Incremental, no LLM merges (ablation) | 79 | 0 | 1.3% | 21 / 21 | 19 / 18 | 96.6% | 98.6% | 97.9% | 33 (33) |
| Batch GraphRAG | 106 | 0 | 36.4% | 39 / 21 | 28 / 18 | 96.6% | 98.6% | 81.5% | 0 |

Relation precision and recall are identical by construction: both graphs come from the same extractions, and the metric maps every node to its true entity before comparing triples, so it measures the extractor, not the merging. The extractor's own errors account for the 3.4% of wrong relations (a listing note whose price and delivery were dropped, "I liked what I saw about X" read as a visit, "Forget Quarry Lofts" turned into a visit and an objection). Current-value accuracy is the share of true single-valued slots whose open values are exactly the true current value; the batch graph keeps 32 slots with two or more conflicting values. Resolution routes in the incremental graph: 38 by chat thread (plus 21 new threads), 119 exact, 10 by rule, 58 new entities, 4 by the LLM.

### Cost of absorbing new episodes (the last quarter of each stream: 32 of 130 episodes)

| strategy | LLM calls | tokens (prompt + output) | LLM seconds | LLM seconds per new episode |
|---|---|---|---|---|
| incremental: fold each new episode in | 34 (32 extractions, 2 merges) | 21,313 | 71 | **2.2** |
| batch: rebuild from scratch | 146 (130 extractions, 16 reports) | 101,826 | 1,106 | 34.6 |
| batch: rebuild, extractions replayed from a cache | 16 (reports only) | 12,558 | 389 | 12.1 |
| hybrid: re-index BM25 and LSA | 0 | 0 | 0 (0.2 s of numpy) | 0 |

The batch rows assume one rebuild for the whole batch of 32. A batch graph as fresh as the incremental one, rebuilt after every episode, would pay a full rebuild each time: 1,106 s per episode, or 389 s with an extraction cache, because community reports depend on the whole graph. One extraction call: median 4.2 s, 657 prompt and 29 output tokens on average; one community report: 24 s for about 109 output tokens. Merge decisions are 1 output token (1.0 to 1.5 s). The incremental graph's deterministic work is under 0.1 s per broker for the 32 episodes. (`results/report.md` shows 2.6 s of "python seconds" for it, because the two merge calls ran inside the resolution step's timer; they are also counted in its LLM columns.)

### Read plainly

- **The incremental graph wins overall, and the margin is not noise** (+12.3 points against hybrid, +13.6 against batch, both intervals above zero). Most of it is aggregation: counting clients who want 2 bedrooms or have a budget above a threshold, and finding the highest budget, needs every client's current value, which the graph's client table gives in one place and eight retrieved messages never do (hybrid 0 of 9). It also answered all 15 single-hop questions.
- **It does not beat hybrid on "now" questions, and its stale-fact rate is the worst result in this repository.** 7 of the 21 answers whose slot had changed stated the old value, against 1 for hybrid. Five of them are multi-hop ("who develops the building Marina Okafor is interested in now?"), two are building prices. In every one checked the graph was right: the superseded edge was closed and marked "superseded: held from ... to ...", the current one marked "current", and the 7B model still took the line listed first. The hybrid context lists messages oldest first, so the newest value is the last thing the model reads, and that ordering helps a small model more than a "current" label does. The graph stores time correctly; its context does not yet make the right reading easy enough for a 7B answerer. This is the first thing to fix (next steps).
- **Batch GraphRAG is hurt by what it lacks, not by extraction.** It has the same facts but no notion of their end: temporal "now" accuracy drops 26.7 points below hybrid (interval excludes zero), and its client table shows "Kate Moreau" and "Katherine Moreau" as two clients with two budgets. Its community reports did not rescue it. Its low wrong-period rate on "as of" questions (8.3%) is not an advantage: it declined 7 of those 12 questions and answered 3 correctly, against 6 correct for the incremental graph.
- **Multi-hop is weak everywhere** (1, 0 and 2 of 12). The owner's example ("which of my clients looking for 3 bedrooms can afford a building delivered in 2027 near Riverside station?") was answered correctly once in 9 tries across the three systems, by the incremental graph. All the facts it needs were in the incremental context (client table, building table); the join across bedrooms, budget, price, delivery year and station is beyond what this model does reliably in one pass. A graph makes the facts available; it does not make a 7B model do a four-way join.
- **Entity resolution is where the incremental graph is clearly better as a structure.** No wrong merge in 80 nodes, both Marinas and both Daniels kept apart, every nickname contact ("Kate Moreau", "Fil Arantes", "Dani Reyes", "Jonah P") joined to the full name, mostly through the chat-thread key. The batch graph's name-only merge split 17 of 21 clients into two or more nodes.
- **The LLM merge step bought nothing here, and cost one duplicate.** It was called 4 times, always for "the Aurora", and was right 3 times; the fourth time it answered "new" and created a stray "Aurora" node. The ablation that simply takes the oldest candidate was right all 4 times, because Aurora Residences is always listed first in this data. That is luck in the data, not a better rule; the step is cheap (1 token, about 1 s) and four calls are too few to judge it.
- **Isolation by construction works, and the contrast shows why it matters.** The same hybrid retriever over one index for all brokers leaked another broker's client budget in 8 of 9 probes; per broker, no system leaked, and all declined.
- **Update cost: the incremental graph is about 15 times cheaper than a rebuild** in model time for this batch of 32 episodes (5 times fewer tokens), and the gap grows with the stream: its cost per episode is constant, a rebuild grows with the corpus. Hybrid search costs nothing to update, so the graph's real price is the one extraction call per message (about 5 s here) that hybrid search never pays.

### Process notes

- **JSON was too expensive.** The first extraction prompt asked for JSON; the model pretty-printed it at about 200 output tokens per listing note and 15 to 39 s per episode, and wrote predicates as "building developer". The line format cut output to 29 tokens on average.
- **A retrieval change made after reading one development answer.** In a 12-call development check, all three systems answered "What was Katherine Moreau's budget on March 21, 2026?" with another client's budget, stated in a message dated exactly March 21. In the graph systems that line came through the BM25 net over all facts, which ran even when the client was linked by full name. The net now runs only when linking finds nothing. The reference run was made after this change, with a fresh cache.
- **The batch graph's "python seconds" and the incremental merge timing** are explained under the cost table; nothing else in the run was measured twice.


## How to run

Requirements: Python 3.11+ (developed on 3.14.8) with numpy and requests, and an Ollama server with `qwen2.5-coder:7b`. The test suite needs neither Ollama nor network.

```bash
# offline tests (scripted LLM with an oracle extractor and answerer, about 2 s)
python3 -m unittest discover -s tests -t .

# the reference run: generate the stream, ingest, build, answer, evaluate, report
python3 -m incgraph run-all

# step by step
python3 -m incgraph synth                 # data/: episodes, facts, entities, questions (seeded, byte-identical)
python3 -m incgraph run                   # every stage, every system; results/
python3 -m incgraph eval                  # the same, replaying the cache; any cache miss is an error
python3 -m incgraph report                # rebuild results/report.md from the result files, no model calls
python3 -m incgraph ask a "What is Katherine Moreau's budget now?" --system incremental_graph
```

Every LLM call goes through a sqlite3 cache (`.cache/llm.sqlite`, not committed) keyed on the full request; `eval` replays the reference run from it in seconds, and deleting it gives an uncached run. Model, retrieval depths, community size, the share of the stream treated as new, and answer length live in `config.toml`.

## Limitations

- **Synthetic, templated stream.** 130 episodes from templates written by the person who wrote the questions, the extraction prompt and the resolver. Real WhatsApp chats are longer, messier, mix several topics per message and refer back across weeks; the confounders here are deliberate and cleaner than real ones. The nickname table and the building-name rules were written knowing the data's naming patterns, even though the Portuguese nicknames were left out of the table on purpose.
- **Small sample.** 81 questions, 9 to 15 per type: a single question moves a per-type accuracy by 7 to 11 points, and most per-type differences in the tables are inside their intervals. Read the per-type rows as direction, not as measurements.
- **One small model does everything.** `qwen2.5-coder:7b` extracts, resolves ambiguous names, writes community reports and answers. Its extraction errors (dropped fields, a "liked what I saw" read as a visit) flow into both graphs equally; a stronger extractor would raise both graph systems, not the hybrid baseline.
- **Batch GraphRAG is a scaled-down reimplementation,** not Microsoft's package: no gleaning passes, no description summarization, Louvain instead of Leiden, no global map-reduce search. It is the indexing recipe, built here on the same extraction as the incremental graph, so the comparison isolates resolution, time and update cost; it does not measure GraphRAG's query-focused summarization on a large corpus, where its community reports earn their cost.
- **No neural embeddings.** The hybrid baseline's dense side is LSA, because the local Ollama refuses `/api/embed` and nothing could be installed. A real embedder would help the baseline mostly on paraphrased names.
- **Isolation is tested, not proven.** Zero leaks for the per-broker systems is a property of the construction (one graph and one index per broker, nothing shared), and the probes confirm the code respects it; they cannot find a leak path the design does not have, such as a shared cache keyed without the broker.
- **Cost is one laptop on a shared server.** Latencies were measured on an Apple M5 with 16 GB of RAM while the same Ollama served a production system, so they vary between runs; token counts are the stable measure. Determinism (temperature 0, seed 42) is per machine; the cache is what makes a rerun exact.

## Next steps

- Render "now" contexts with only the current edges (history behind a separate heading, or only for "as of" questions), and repeat the run: that tests whether the 33% stale rate is a context-layout problem, as the inspected cases suggest.
- Answer list, count and fit questions with code over the graph (filter clients by current bedrooms and budget, join buildings by station and delivery year) and let the model only phrase the result; multi-hop accuracy should not depend on a 7B model doing a four-way join.
- A real stream: anonymized broker chats, longer and messier, with references across weeks, and a second annotator for the ground truth.
- More ambiguous mentions, so the LLM merge step can be measured on more than 4 decisions, and a neural embedder for the hybrid baseline when one can be installed.

## Layout

```
incgraph/
  cli.py          python -m incgraph <synth|run|eval|report|run-all|ask>
  synth.py        seeded fictional brokerage: entities, episode stream, facts with validity intervals
  questions.py    typed gold questions per broker, derived from the ground truth
  llm.py          Ollama over requests (pinned keep_alive and options, back-off), scripted fake, cache, ledger
  extract.py      per-episode extraction prompt, line parser, value normalization
  values.py       amounts, months, counts and names: parsing and normalizing
  graph.py        per-broker graph in plain dicts; temporal edges (valid_from / valid_to) or batch edges
  resolve.py      entity resolution: thread, exact, rules, LLM for ambiguous merges
  incremental.py  system 3: fold one episode into the graph
  batch.py        system 2: extract all, merge by name, Louvain, community reports
  retrieve.py     system 1 (BM25 + LSA + RRF over episodes) and graph retrieval (link, 2 hops, BM25, tables)
  bm25.py         Okapi BM25 in numpy (from hybrid-search-rag)
  fusion.py       Reciprocal Rank Fusion (from hybrid-search-rag)
  text.py         tokenizer and "I don't know" detector (from hybrid-search-rag)
  answer.py       the one answer prompt; answer line and citations
  scoring.py      answer scoring against gold atoms; paired bootstrap
  graphq.py       graph quality against the ground truth
  pipeline.py     run everything once, derive every table
  report.py       results/report.md
data/             generated stream, ground truth and gold questions
results/          artifacts of the reference run (report.md first; graphs/, answers, ledger)
tests/            unittest suite, offline
```

## License

MIT, see `LICENSE`.
