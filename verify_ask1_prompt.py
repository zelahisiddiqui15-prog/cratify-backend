"""ASK1 — the current ask decides, what is already in the song is said once, and
half an answer is never silent.

Measured on Zee's five real asks, 2026-10-01, project "in the dark" (Bb Major,
128 BPM), on the build installed that evening:

  - "need some drum loops for this song" came back led by a 2.2-second
    "Illenium Style Fill" from RANK 44 OF 50, because the producer had asked
    for an Illenium fill earlier in the thread. Nothing ranked that row up —
    the conversation did, in the prompt. Rule 1's prompt half is here.
  - three of the four searches LED with a file already dragged into that very
    project, and no reply mentioned it. Rule 6 is here.
  - "find me chords that fit" returned eight chords, none in the project's key,
    and `unmet` came back EMPTY. The library holds 232 chord rows in Bb and 82
    at 120-136 BPM and ZERO with both, so the ask was unanswerable whole and
    nobody said so. The relaxation rule is here.

The renderer is EXECUTED, not grepped, for the same reason FIT2's guard is: a
flag that does not reach the row is a flag the model cannot read.
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
renderer_src = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "SEARCH_SYSTEM_DIRECTIVES":
                directives = node.value.value
    if isinstance(node, ast.FunctionDef) and node.name == "render_search_candidates":
        renderer_src = ast.get_source_segment(src, node)

head("0. the pieces exist")
ok("SEARCH_SYSTEM_DIRECTIVES found", directives is not None)
ok("render_search_candidates found", renderer_src is not None)
if directives is None or renderer_src is None:
    print("\n  ASK1 prompt: RED — cannot run without the real pieces")
    sys.exit(1)

# ── 1. rule 1's prompt half ──────────────────────────────────────────
head("1. rule 1 — the CURRENT ask decides kind and category")
ok("the rule is stated", "THE CURRENT ASK DECIDES" in directives)
ok("…earlier turns are background", "Earlier turns are BACKGROUND" in directives)
ok("…and never a reason to return something the ask did not call for",
   "never a reason to return a file the current ask did not call for" in directives)
ok("the tie-break is explicit", "THE CURRENT ASK WINS" in directives)
ok("…including the consequence of honouring it", "return fewer" in directives)
ok("the measured case is on the record, so the rule cannot be read as taste",
   "Illenium Style Fill" in directives and "rank 44 of 50" in directives)
# The older CONTEXT rule says a bare follow-up INHERITS. Both are true and the
# new rule must not have deleted the old one — that is how "okay how about
# serum?" keeps working.
ok("the inheritance rule it sits beside is intact",
   "A follow-up that names NO category inherits the category of the previous turn" in directives)
ok("…and the new rule says so itself rather than contradicting it",
   "This is the opposite failure to the CONTEXT rule above, and both are real" in directives)

# ── 2. rule 6 ────────────────────────────────────────────────────────
head("2. rule 6 — already in this song, said once, never led with")
ok("the line is named", "ALREADY IN THIS SONG" in directives)
ok("do not lead with one", "DO NOT LEAD WITH ONE" in directives)
ok("…but it is a preference, not a ban",
   "this is a preference, not a ban" in directives)
ok("mention them in ONE line", "MENTION THEM IN ONE LINE" in directives)
ok("…with the wording Zee asked for",
   "you already have the ESW kit loop in this song" in directives)
ok("it is kept distinct from FIX7's RECENTLY TRIED",
   "NOT the same line as RECENTLY TRIED" in directives)
ok("…and FIX7's own line is untouched",
   "It is NOT a record of usage" in directives
   and '"in your track"' in directives)

# ── 3. the relaxation ────────────────────────────────────────────────
head("3. nothing matches both key and tempo")
ok("the context line is named", "NOTHING HERE MATCHES BOTH YOUR KEY AND YOUR TEMPO" in directives)
ok("relax to one half rather than pretend", "RELAX TO ONE HALF" in directives)
ok("and name the dropped half in `unmet`", "NAME THE HALF YOU DROPPED IN `unmet`" in directives)
ok("…with the user's own words asked for", "in their words" in directives)
ok("the measurement is recorded", "232 chord rows" in directives)
ok("…and it is called what it is", "lie by omission" in directives)

# ── 4. the row flag, executed ────────────────────────────────────────
head("4. rule 6's flag reaches the row the model reads")
ns = {}
exec(renderer_src, ns)  # noqa: S102 — running the REAL renderer is the point
render = ns["render_search_candidates"]
ROW = {"id": 7, "meta_text": "ESW Kit 03 - Drums.wav — A drum loop. — 26.5 s"}
plain, _ = render([dict(ROW)])
flagged, _ = render([{**ROW, "in_song": True}])
ok("a row with no flag says nothing about the song", "already in this song" not in plain, repr(plain))
ok("a flagged row says it", "already in this song" in flagged, repr(flagged))
ok("…and still leads with what it always led with", flagged.startswith(f"[{ROW['id']}] {ROW['meta_text']}"))
for junk in (False, None, "yes", 1):
    t, _ = render([{**ROW, "in_song": junk}])
    ok(f"in_song={junk!r} is not a flag", "already in this song" not in t)
# Both facts on one row, in a fixed order, so the line cannot read as two rows.
both, seen = render([{**ROW, "in_song": True, "fit": {"total": 0.02, "why": "128 ✓"}}])
ok("the flag and the fit coexist on one line", "already in this song" in both and "fit +0.0200" in both, repr(both))
ok("…and the fit is still what the fit directive describes", seen is True)

# ── 5. the rules it sits beside ──────────────────────────────────────
head("5. nothing older was trampled")
for label, text in [
    ("REASON1's unmet rule", "put it in `unmet` in their own words"),
    ("the CATEGORY rule", "your picks MUST be of that category"),
    ("the typed-key rule", "A KEY THE USER TYPED OUTRANKS TEMPO, GENRE AND THE PROJECT'S KEY"),
    ("the MIDI limit", "THE MIDI LIMIT IS MANDATORY"),
    ("plain text only", "PLAIN TEXT ONLY"),
    ("the no-ears rule", "YOU CANNOT HEAR THE AUDIO"),
]:
    ok(f"{label} still holds", text in directives)

print(
    "\n  ASK1 prompt: the current ask decides, the song is named once, and half an answer says so — "
    + ("green" if not failures else f"RED ({failures})")
)
sys.exit(1 if failures else 0)
