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

# ── rulings 1 and 2, in words ────────────────────────────────────────
head("rule 5 — the role word's meaning is STATED, not only ranked")
ok("drum loops are full loops", '"DRUM LOOPS" / "a drum loop" / "a break" = FULL LOOPS' in directives)
ok("…four seconds or longer", "four seconds or longer" in directives)
ok("…never a fill, tom, hit or one-shot",
   "NEVER a fill, a tom, a hit or a one-shot" in directives)
ok("…and the tag is named as untrustworthy, with the measurement",
   "164 rows typed `sample_type='loop'` are under three seconds" in directives)
ok("…including the case that beat the ranking three times",
   "even when it sits at exactly the project tempo" in directives)
ok("chords put keys presets before pads", '"CHORDS" = chord loops and chord MIDI first' in directives
   and "Pads come last" in directives)
ok("a structural ask is held to the same standard", "A STRUCTURAL ASK" in directives)
ok("and honouring it may mean fewer picks", "RETURN FEWER" in directives)

head("build asks — any category, two seconds, audio first")
ok("any category qualifies if the NAME or TAGS carry the word",
   "ANY CATEGORY QUALIFIES as long as the file's own name or tags carry that word" in directives)
ok("…bass risers named as an example", "a bass riser is build material" in directives)
ok("…and the Drums/FX restriction is explicitly lifted",
   "Do not restrict yourself to Drums and FX" in directives)
ok("two seconds minimum", "at least TWO SECONDS" in directives)
ok("audio first", "AUDIO COMES FIRST" in directives)
ok("…presets only below three audio matches",
   "if fewer than three audio rows in the candidate list carry the structural word" in directives)
ok("…and then unmet says so", "`unmet` must say so" in directives)

# ── ruling 3, executed ───────────────────────────────────────────────
head("ruling 3 — one file never appears twice in a reply")
import importlib.util as _ilu  # noqa: E402
# The dedupe is inline in the handler, so it is exercised through the HANDLER's
# source rather than a function: the loop is lifted and run on real pick lists.
fn = next(n for n in ast.walk(tree)
          if isinstance(n, ast.FunctionDef) and n.name in ("search", "api_search", "search_endpoint"))
body = ast.get_source_segment(src, fn)
start = body.index("_seen_ids = set()")
end = body.index("picks_raw = _deduped") + len("picks_raw = _deduped")
snippet = "\n".join(l[4:] if l.startswith("    ") else l for l in body[start:end].split("\n"))
def dedupe(picks):
    ns2 = {"picks_raw": list(picks), "print": lambda *a, **k: None}
    exec(snippet, ns2)  # noqa: S102 — the REAL loop, lifted
    return ns2["picks_raw"]
ok("a clean crate is untouched",
   [p["id"] for p in dedupe([{"id": 1}, {"id": 2}, {"id": 3}])] == [1, 2, 3])
ok("a duplicate is dropped and the FIRST one survives",
   [p["id"] for p in dedupe([{"id": 1, "reason": "first"}, {"id": 2}, {"id": 1, "reason": "second"}])] == [1, 2]
   and dedupe([{"id": 1, "reason": "first"}, {"id": 1, "reason": "second"}])[0]["reason"] == "first")
ok("the measured case — one file twice in one crate — collapses to one",
   len(dedupe([{"id": 9}, {"id": 9}])) == 1)
ok("three of the same collapse to one", len(dedupe([{"id": 9}, {"id": 9}, {"id": 9}])) == 1)
ok("a pick with no id is kept rather than silently eaten",
   len(dedupe([{"reason": "no id"}, {"reason": "also none"}])) == 2)
ok("…and an empty list stays empty", dedupe([]) == [])

# ── ruling 4, executed ──────────────────────────────────────────────
head("ruling 4 — the song note is appended in CODE, not asked for")
ns = {"_re": __import__("re")}
for name in ("_ALREADY_SAID", "song_note", "needs_song_note"):
    node = next((n for n in ast.walk(tree)
                 if (isinstance(n, ast.FunctionDef) and n.name == name)
                 or (isinstance(n, ast.Assign) and any(
                     isinstance(t, ast.Name) and t.id == name for t in n.targets))), None)
    ok(f"{name} exists", node is not None)
    if node is not None:
        exec(ast.get_source_segment(src, node), ns)  # noqa: S102
note, needs = ns["song_note"], ns["needs_song_note"]
ok("one file reads as one clause", note(["ESW Kit 03.wav"]) == "You already have ESW Kit 03.wav in this song.")
ok("two files read as two", note(["a.wav", "b.wav"]) == "You already have a.wav and b.wav in this song.")
ok("more than two are counted, not listed",
   note(["a.wav", "b.wav", "c.wav", "d.wav"]) == "You already have a.wav and b.wav, and 2 more in this song.")
ok("nothing in the song means no line", note([]) is None and note(["", "  "]) is None)
ok("it fires when the model said nothing", needs("Here are some drum loops.", ["a.wav"]) is True)
for said in ("You already have the ESW kit in this song.", "you already got that one",
             "that's in your song already", "the one you dragged in earlier"):
    ok(f"…and NOT when the model already said it ({said[:28]}…)", needs(said, ["a.wav"]) is False)
ok("…and never when there is nothing to say", needs("Here you go.", []) is False)

# AND THE NAMES IT IS GIVEN. Loosening the handler's filter from
# `in_song is True` to `is not None` left this section green too: every
# assertion fed `song_note` a list someone else had built. So the real
# comprehension is lifted and run, the same way the dedupe loop is.
_i0 = body.index("_in_song_names = [")
_i1 = body.index("]", body.index('c.get("in_song")')) + 1
_namesrc = "\n".join(l[4:] if l.startswith("    ") else l for l in body[_i0:_i1].split("\n"))
def names_from(cands):
    ns3 = {"candidates": cands}
    exec(_namesrc, ns3)  # noqa: S102 — the REAL comprehension
    return ns3["_in_song_names"]
M = "ESW Kit 03 - Drums.wav — A drum loop. — 26.5 s"
ok("a flagged candidate contributes its filename, not its whole row",
   names_from([{"id": 1, "meta_text": M, "in_song": True}]) == ["ESW Kit 03 - Drums.wav"])
ok("an unflagged candidate contributes nothing",
   names_from([{"id": 1, "meta_text": M}]) == [])
for junk in (False, None, "yes", 1, 0):
    ok(f"in_song={junk!r} does not count as being in the song",
       names_from([{"id": 1, "meta_text": M, "in_song": junk}]) == [])
ok("two flagged rows give two names, in list order",
   names_from([{"id": 1, "meta_text": M, "in_song": True},
               {"id": 2, "meta_text": "b.wav — x", "in_song": True}]) == ["ESW Kit 03 - Drums.wav", "b.wav"])

# THE PREDICATE IS NOT THE FEATURE. Disabling the append in the handler
# (`if False:`) left this section GREEN, because every assertion above tests the
# two functions and none tested that anything CALLS them. So: the wiring, read
# off the AST, and ordered against the mention-span validation that must see the
# final text.
uses = lambda nid: [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == nid]
ok("the handler calls needs_song_note", len(uses("needs_song_note")) == 1, str(uses("needs_song_note")))
ok("…and song_note, exactly once each", len(uses("song_note")) == 1)
ok("…inside an `if`, not unconditionally",
   any(isinstance(n, ast.If) and "needs_song_note" in (ast.get_source_segment(src, n.test) or "")
       for n in ast.walk(fn)))
ok("…and it APPENDS to reply_val rather than replacing it",
   "reply_val.rstrip() + \" \" + song_note(" in body)
# The MIDI caveat and the song note must both land before the spans are checked,
# or a mention span can point into text the client never receives.
i_note = uses("needs_song_note")[0] if uses("needs_song_note") else 10 ** 9
i_spans = min((n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == "mentions"), default=10 ** 9)
ok("…before the mention spans are validated against the final reply", i_note < i_spans,
   f"note at {i_note}, spans at {i_spans}")

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
