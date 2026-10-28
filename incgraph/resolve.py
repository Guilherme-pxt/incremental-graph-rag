"""Entity resolution for the incremental graph: deterministic first, the LLM only for real ambiguity.

A name the extractor returned is matched to an existing node, in this order:

1. **Chat thread.** In a chat, a client name compatible with the chat's
   contact (same surname, or a compatible first name with no surname) is the
   thread's client. A WhatsApp thread is a phone number, the strongest key a
   broker has.
2. **Exact name or alias** of the same type, normalized (case, punctuation,
   a leading "the", honorifics).
3. **Rules.** People: compatible first names (equal, a nickname from a small
   table of common English nicknames, or a prefix of 3+ letters) and the same
   surname, or a surname initial ("Jonah P."). Buildings: every word of the
   mention equals or starts a word of the name ("Aurora Res." is Aurora
   Residences; "the Lofts" is Quarry Lofts). Developers, neighborhoods and
   stations: the same words, with "station" ignored.
4. **The LLM** decides only when the rules leave more than one candidate
   ("the Aurora": Aurora Residences or Aurora Park?) or only a weak one (same
   surname with a first name the table does not know, a bare first name). It
   sees the message, the chat context and what the graph knows about each
   candidate, and answers with one digit; 0 means "someone or something new".
5. Otherwise a new node.

A surface form becomes an alias of a node only when it pointed to exactly one
candidate, so an ambiguous form like "the Aurora" is never learned as a name.
Every decision is logged with its route, for the graph-quality tables.
"""

from __future__ import annotations

from .extract import episode_text
from .llm import choice_probs
from .values import norm_name

NICKNAMES = {
    "kate": ["katherine", "catherine", "kathryn"], "katie": ["katherine", "catherine"], "kathy": ["katherine"],
    "will": ["william"], "bill": ["william"], "billy": ["william"], "liam": ["william"],
    "bob": ["robert"], "rob": ["robert"], "robbie": ["robert"], "sam": ["samuel", "samantha"],
    "liz": ["elizabeth", "elisabeth"], "beth": ["elizabeth", "elisabeth"], "lizzie": ["elizabeth", "elisabeth"],
    "dani": ["daniel", "daniela", "danielle"], "dan": ["daniel"], "danny": ["daniel"], "jim": ["james"],
    "jimmy": ["james"], "mike": ["michael"], "tom": ["thomas"], "alex": ["alexander", "alexandra"],
    "nick": ["nicholas"], "chris": ["christopher", "christina", "christine"], "jen": ["jennifer"],
    "jenny": ["jennifer"], "joe": ["joseph"], "matt": ["matthew"], "tony": ["anthony"], "ben": ["benjamin"],
    "nate": ["nathan", "nathaniel"], "pat": ["patrick", "patricia"], "sue": ["susan"], "meg": ["margaret"],
    "peggy": ["margaret"], "ted": ["edward", "theodore"], "ed": ["edward"], "andy": ["andrew"],
    "abby": ["abigail"], "steve": ["steven", "stephen"], "greg": ["gregory"], "jon": ["jonathan"],
}
GENERIC_BUILDING_WORDS = {"tower", "building", "project", "condo", "residence"}


def first_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    if b in NICKNAMES.get(a, []) or a in NICKNAMES.get(b, []):
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 3 and long_.startswith(short)


def last_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    return (len(a) == 1 and b.startswith(a)) or (len(b) == 1 and a.startswith(b))


def person_level(mention: str, name: str) -> int:
    """3 = same person by rule (first compatible, same surname); 2 = first compatible and a surname initial;
    1 = weak (bare first name, or same surname with a first name the rules cannot connect); 0 = no."""
    m, n = norm_name(mention).split(), norm_name(name).split()
    if not m or not n:
        return 0
    mf, ml = m[0], (m[-1] if len(m) > 1 else None)
    nf, nl = n[0], (n[-1] if len(n) > 1 else None)
    fc = first_compatible(mf, nf)
    if ml and nl:
        if ml == nl and fc:
            return 3 if len(ml) > 1 else 2
        if last_compatible(ml, nl) and fc:
            return 2
        if ml == nl and len(ml) > 1:
            return 1
        return 0
    return 1 if fc else 0


def _fuller(surface: str, name: str) -> bool:
    """A two-word name with no initials, replacing a name that has an initial or a single word, same person."""
    s, n = norm_name(surface).split(), norm_name(name).split()
    full = len(s) >= 2 and min(map(len, s)) > 1
    partial = len(n) < 2 or min(map(len, n)) == 1
    return full and partial and person_level(surface, name) >= 2


def building_match(mention: str, name: str) -> bool:
    m = [w for w in norm_name(mention).split() if w not in GENERIC_BUILDING_WORDS]
    n = [w for w in norm_name(name).split() if w not in GENERIC_BUILDING_WORDS]
    if not m or not n:
        return False
    return all(any(t == w or (len(w) >= 3 and t.startswith(w)) for t in n) for w in m)


def place_match(mention: str, name: str) -> bool:
    strip = lambda s: [w for w in norm_name(s).split() if w not in {"station", "metro", "neighborhood"}]
    m, n = strip(mention), strip(name)
    return bool(m) and m == n


class Resolver:
    def __init__(self, graph, llm=None, broker_name: str = "", use_llm: bool = True, top_logprobs: int = 10):
        self.g = graph
        self.llm = llm
        self.broker_name = broker_name
        self.use_llm = use_llm and llm is not None
        self.top_logprobs = top_logprobs
        self.log: list[dict] = []

    def resolve(self, surface: str, etype: str, ep: dict, context: list[dict]) -> str:
        nid, route, cands = self._resolve(surface, etype, ep, context)
        node = self.g.nodes[nid]
        node["mentions"].append([ep["id"], surface])
        if route in ("thread", "exact", "rule", "new", "thread-new") or (route == "llm" and len(cands) == 1):
            self.g.add_alias(nid, surface)
        if etype == "client" and route.startswith("thread") and ep.get("contact"):
            self.g.add_alias(nid, ep["contact"])
        if etype == "client" and _fuller(surface, node["name"]):
            old = node["name"]
            node["name"] = surface  # keep the fullest name seen as the display name ("Jonah Pellegrini", not "Jonah P")
            node["aliases"] = sorted(set(node["aliases"] + [old]) - {surface})
        self.log.append({"episode": ep["id"], "surface": surface, "type": etype, "node": nid, "route": route,
                         "candidates": cands})
        return nid

    def _resolve(self, surface: str, etype: str, ep: dict, context: list[dict]) -> tuple[str, str, list[str]]:
        g = self.g
        thread = ep.get("thread") if etype == "client" else None
        if thread and ep.get("contact") and person_level(surface, ep["contact"]) >= 1:
            if thread in g.thread_index:
                return g.thread_index[thread], "thread", []
            nid, route, cands = self._by_name(surface, etype, ep, context)
            if nid is None:
                return g.add_node(etype, surface, ep["id"], thread=thread), "thread-new", cands
            g.nodes[nid]["thread"] = thread
            g.thread_index[thread] = nid
            return nid, route, cands
        nid, route, cands = self._by_name(surface, etype, ep, context)
        if nid is None:
            nid = g.add_node(etype, surface, ep["id"])
        return nid, route, cands

    def _by_name(self, surface: str, etype: str, ep: dict, context: list[dict]):
        """(node id, route, candidates); node id None means 'a new entity' (the caller creates it)."""
        g = self.g
        exact = sorted(g.lookup(etype, surface))
        if len(exact) == 1:
            return exact[0], "exact", exact
        if len(exact) > 1:
            return self._ask(surface, etype, exact, ep, context)
        pool = g.of_type(etype)
        if etype == "client":
            levels = {nid: max(person_level(surface, n) for n in g.names(nid)) for nid in pool}
            for lvl in (3, 2):
                top = sorted(nid for nid, v in levels.items() if v == lvl)
                if len(top) == 1:
                    return top[0], "rule", top
                if len(top) > 1:
                    return self._ask(surface, etype, top, ep, context)
            weak = sorted(nid for nid, v in levels.items() if v == 1)
            if weak:
                return self._ask(surface, etype, weak, ep, context)
        elif etype == "building":
            cands = sorted(nid for nid in pool if any(building_match(surface, n) for n in g.names(nid)))
            if len(cands) == 1:
                return cands[0], "rule", cands
            if len(cands) > 1:
                return self._ask(surface, etype, cands, ep, context)
        else:
            cands = sorted(nid for nid in pool if any(place_match(surface, n) for n in g.names(nid)))
            if len(cands) == 1:
                return cands[0], "rule", cands
        return None, "new", []

    # ------------------------------------------------------------ the LLM, for ambiguous merges

    def describe(self, nid: str) -> str:
        g = self.g
        node = g.nodes[nid]
        facts = []
        for e in g.edges_of(nid):
            if g.temporal and e["valid_to"] is not None:
                continue
            if e["src"] == nid:
                facts.append(f"{e['pred']} {g.obj_text(e)}")
            else:
                facts.append(f"{g.nodes[e['src']]['name']} {e['pred']} it")
        names = ", ".join(g.names(nid))
        return f"{names}: " + ("; ".join(facts[:8]) if facts else "no facts yet")

    def merge_messages(self, surface: str, etype: str, cands: list[str], ep: dict, context: list[dict]) -> list[dict]:
        lines = [f'A {etype} is mentioned as "{surface}" in this message:']
        if context:
            lines.append("(earlier in the same chat)")
            lines.extend(episode_text(c, self.broker_name) for c in context)
        lines.append(episode_text(ep, self.broker_name))
        lines.append(f"\nKnown {etype}s it could be:")
        lines.extend(f"{i}. {self.describe(nid)}" for i, nid in enumerate(cands, start=1))
        lines.append(f"0. none of these: a different {etype}")
        lines.append(f'\nWhich one is "{surface}"? Reply with the number only.')
        return [{"role": "user", "content": "\n".join(lines)}]

    def _ask(self, surface: str, etype: str, cands: list[str], ep: dict, context: list[dict]):
        cands = cands[:9]
        if not self.use_llm:
            return cands[0], "first", cands  # ablation: take the oldest candidate, no LLM
        res = self.llm.chat(self.merge_messages(surface, etype, cands, ep, context), max_tokens=1,
                            tag="inc:merge", ref=ep["id"], top_logprobs=self.top_logprobs)
        opts = [str(i) for i in range(len(cands) + 1)]
        probs = choice_probs(res.top_logprobs, opts, res.text)
        pick = max(opts, key=lambda o: (probs[o], -int(o)))
        if pick == "0":
            return None, "llm-new", cands
        return cands[int(pick) - 1], "llm", cands
