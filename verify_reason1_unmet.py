"""REASON1 — the model must be able to say what it could not satisfy, and
must be told to say it when it trades one constraint for another.

THE CASE THIS EXISTS FOR, measured on Zee's library 2026-09-26. The ask
"Give me a punchy kick drum" returned eight kicks at 0.78-0.92 confidence and
NOT ONE was tagged punchy — because the only three punchy kicks in the library
are Nexus PRESETS. Retrieval had done its job: 22 punchy rows made the fifty,
including all three of those presets. The model preferred the noun over the
adjective and samples over presets, which is defensible, and said nothing,
which is not. The producer saw eight confident answers with no hint that half
the ask had been dropped.

Zee's ruling: "the ask decides. 'Kick drum' with no file kind named means
audio first; the model may prefer samples — but it must SAY so via `unmet`
('the only punchy kicks I have are Nexus presets'). Never silently."

Lifted from the REAL source, so deleting the directive or the schema field
fails here.
"""
import ast
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
src = open(SRC).read()
tree = ast.parse(src)

failures = 0


def ok(label, cond, detail=""):
    global failures
    if not cond:
        failures += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))


def head(s):
    print(f"\n  {s}:")


directives = None
search_tool = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "SEARCH_SYSTEM_DIRECTIVES":
                directives = ast.literal_eval(node.value)
            if isinstance(t, ast.Name) and t.id in ("SEARCH_TOOL", "SEARCH_TOOLS"):
                try:
                    search_tool = ast.literal_eval(node.value)
                except ValueError:
                    search_tool = None
if directives is None:
    print("  SEARCH_SYSTEM_DIRECTIVES not found — the prompt moved")
    sys.exit(1)

# ── 1. the schema can carry the answer ───────────────────────────────
head("1. `unmet` exists, and is REQUIRED so silence is not ambiguous")
ok('the schema declares an "unmet" array', '"unmet": {' in src)
ok("...of strings", '"items": {"type": "string"}' in src)
ok(
    "...and it is REQUIRED, so an empty list means the picks fit",
    '"required": ["picks", "reply", "filters_used", "mentions", "unmet"]' in src,
)
ok(
    "the response returns it, normalised to non-empty strings",
    'unmet_out = (' in src and 'isinstance(u, str) and u.strip()' in src,
)
ok(
    "...and a malformed value reads as nothing unmet rather than failing a paid search",
    'if isinstance(unmet_raw, list)' in src and 'else []' in src,
)

# ── 2. the prompt tells it WHEN ──────────────────────────────────────
head("2. the prompt names the trade, and names it with the case that earned it")
ok("audio comes first when no file kind is named", "AUDIO COMES FIRST" in directives)
ok("...the model MAY prefer samples", "You MAY prefer samples" in directives)
ok("...BUT MUST SAY SO", "BUT YOU MUST SAY SO" in directives)
ok(
    "the measured case is in the prompt, not paraphrased away",
    "the only three punchy kicks in the library are Nexus PRESETS" in directives
    or "only punchy kicks I have are Nexus presets" in directives,
)
ok(
    "...including the part that makes it a lesson: the silence, not the choice",
    "Both readings were defensible; the silence was not." in directives,
)
ok(
    "the rule is general, not preset-specific",
    "not just presets" in directives
    and all(w in directives for w in ("a key you could not match", "a tempo nothing sits at", "a mood nothing carries")),
)
ok("it must go in `unmet`, in the user's own words", "put it in `unmet` in their own words" in directives)
ok(
    "...and near-misses must not be dressed up as the answer",
    "do not present the near-misses as though they were what was asked for" in directives,
)

# ── 3. the older rules it must not have trampled ─────────────────────
head("3. the directives it sits beside are intact")
ok("CATEGORY still governs picks", "your picks MUST be of that category" in directives)
ok("the reply shape is still stated once", "THE SHAPE OF THE REPLY IS SPECIFIED HERE AND NOWHERE ELSE" in directives)
ok("plain text only still holds", "PLAIN TEXT ONLY" in directives)
ok("and it still cannot claim to hear audio", "YOU CANNOT HEAR THE AUDIO" in directives)

print(
    "\n  REASON1 unmet: the model can say what it could not satisfy, and is told to — "
    + ("green" if not failures else f"RED ({failures})")
)
sys.exit(1 if failures else 0)
