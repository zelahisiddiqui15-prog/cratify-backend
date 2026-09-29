"""FIT2 — the model can read the fit, the control cannot, and the ask still wins.

RANK2 computed an explicit fit for all fifty candidates and ORDERED them by it.
The model never saw the number: it received `[id] filename — description —
duration` and a list whose order encoded a judgement it had no way to read, so
it re-decided from the text. Measured on q01, the fit had the afro loops roughly
2:1 above the disco/funk breaks (0.0316-0.0326 against 0.0131-0.0174) and the
model took the breaks anyway.

Zee: "send each candidate's fit components to the model with its row
('afro ✓ · 150 ✓ · Gm ✓', fit 0.032) and tell it to prefer higher fit unless the
ask says otherwise, naming the trade in `unmet`."

THE PROPERTY THIS GUARD EXISTS FOR is the one the MEASUREMENT depends on, not
the feature. The A/B runs two desktop builds against ONE backend. The pre-FIT2
arm sends no `fit` key and MUST receive the prompt production serves today,
byte for byte — otherwise the control is not a control, and the installed app
(which predates FIT2) would be told about a field its rows do not carry. So:

  1. no `fit` in the payload  ->  the rendered list is exactly `[id] meta_text`,
     and NO fit directive is sent at all.
  2. `fit` in the payload     ->  the number and its components ride on the row,
     and the directive that explains them is sent.
  3. the CACHED block is byte-identical either way, so the two arms cannot
     differ on cache pricing.

The renderer is EXECUTED, not grepped: the real function is lifted out of
server.py and run, so a change to what it emits fails here.
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


# ── lift the real pieces out of the real file ─────────────────────────
directives = None
fit_directives = None
renderer_src = None
search_fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "SEARCH_SYSTEM_DIRECTIVES":
                directives = node.value.value
            if isinstance(t, ast.Name) and t.id == "FIT_DIRECTIVES":
                fit_directives = node.value.value
    if isinstance(node, ast.FunctionDef) and node.name == "render_search_candidates":
        renderer_src = ast.get_source_segment(src, node)
    if isinstance(node, ast.FunctionDef) and node.name in ("search", "api_search", "search_endpoint"):
        search_fn = node

head("0. the pieces exist")
ok("SEARCH_SYSTEM_DIRECTIVES found", directives is not None)
ok("FIT_DIRECTIVES found", fit_directives is not None)
ok("render_search_candidates found", renderer_src is not None)
if renderer_src is None or fit_directives is None or directives is None:
    print("\n  FIT2 prompt: RED — cannot run without the real pieces")
    sys.exit(1)

ns = {}
exec(renderer_src, ns)  # noqa: S102 — this is the point: run the REAL renderer
render = ns["render_search_candidates"]

ROW = {"id": 4242, "meta_text": "afro_perc_150_Gm.wav — Afro percussion loop. In Gm key. — 4.1 s"}
PLAIN = f"[{ROW['id']}] {ROW['meta_text']}"

# ── 1. THE CONTROL: a pre-FIT2 payload renders what it renders today ──
head("1. the pre-FIT2 arm's prompt is untouched")
text_a, seen_a = render([dict(ROW)])
ok("no `fit` key -> the line is exactly `[id] meta_text`", text_a == PLAIN, repr(text_a[:80]))
ok("no `fit` key -> fit_seen is False", seen_a is False, repr(seen_a))
ok("...so nothing about fit reaches the control", "fit" not in text_a.lower())
# A `fit` that is not a usable number must also read as absent, not as a
# half-rendered row: a client on an older build, or a malformed field, gets the
# control's prompt rather than a directive about a number that is not there.
for junk in (None, "0.032", {}, {"why": "afro ✓"}, {"total": None}):
    t, sn = render([{**ROW, "fit": junk}])
    ok(f"a fit of {junk!r} reads as absent", t == PLAIN and sn is False)

# ── 2. THE ARM: the fit rides with the row ────────────────────────────
head("2. the FIT2 arm carries the number and its components")
text_b, seen_b = render([{**ROW, "fit": {"total": 0.032, "why": "afro ✓ · 150 ✓ · Gm ✓"}}])
ok("fit_seen is True", seen_b is True)
ok("the row still leads with what it always led with", text_b.startswith(PLAIN), repr(text_b[:60]))
ok("the total is on the row, signed", "+0.0320" in text_b, repr(text_b[-40:]))
ok("the components are on the row", "afro ✓ · 150 ✓ · Gm ✓" in text_b)
neg, _ = render([{**ROW, "fit": {"total": -0.014, "why": "128 ✗"}}])
ok("a clash renders as a clash, not as a bonus", "-0.0140" in neg and "128 ✗" in neg, repr(neg[-30:]))
empty, _ = render([{**ROW, "fit": {"total": 0.0, "why": ""}}])
ok("a row with nothing fired says so, not `()`", "no component matched" in empty and "()" not in empty)
ok("the cap is still fifty rows", len(render([dict(ROW, id=i) for i in range(80)])[0].split("\n")) == 50)

# ── 3. what the directive actually tells the model ───────────────────
head("3. prefer higher fit, unless the ask says otherwise, and name the trade")
ok("it explains what the number is", "computed for THIS song before you saw the list" in fit_directives)
ok("PREFER HIGHER FIT", "PREFER HIGHER FIT" in fit_directives)
ok("...as a tie-break between candidates that both answer the ask",
   "both answer the ask" in fit_directives)
ok("the ask outranks it", "UNLESS THE ASK SAYS OTHERWISE" in fit_directives
   and "The user's words outrank this number every time" in fit_directives)
ok("...with the cases named", all(w in fit_directives for w in ("different key", "half tempo")))
ok("a hard constraint the user named still wins",
   "a key they typed" in fit_directives and "a category they asked for" in fit_directives)
ok("going low on purpose goes in `unmet`",
   "SAY SO IN `unmet`" in fit_directives and "name the trade in their own words" in fit_directives)
ok("the ✗ is explained as a clash that cost the row", "a clash that cost the row" in fit_directives)
ok("and the number never reaches the user's reply",
   "Never mention the number itself" in fit_directives)

# ── 4. the cached prefix is identical in both arms ────────────────────
head("4. the arms cannot differ on cache pricing")
ok("the fit directive is NOT inside the cached block",
   "PREFER HIGHER FIT" not in directives and "FIT_DIRECTIVES" not in directives)
ok("the cached block still ends by handing over the candidate list",
   directives.rstrip().endswith("(pre-ranked by similarity, ID in brackets):"))

head("5. the handler sends them in that order")
ok("the search handler was found", search_fn is not None)
if search_fn is not None:
    body = ast.get_source_segment(src, search_fn)
    # The ONE ordering that matters: instructions before the data they describe.
    # Positions come from the AST too, for the same reason the count does: the
    # first version compared body.find("FIT_DIRECTIVES") against the candidate
    # append, and my own comment ("See FIT_DIRECTIVES above") sits above both —
    # so swapping the two appends left this GREEN. A comment cannot satisfy a
    # lineno.
    line_of = lambda nid: next((n.lineno for n in ast.walk(search_fn)
                                if isinstance(n, ast.Name) and n.id == nid), None)
    last_of = lambda nid: max((n.lineno for n in ast.walk(search_fn)
                               if isinstance(n, ast.Name) and n.id == nid), default=None)
    i_fit, i_cand = line_of("FIT_DIRECTIVES"), last_of("candidates_text")
    ok("the renderer is what builds the list",
       "render_search_candidates(candidates)" in body)
    # AST, not a substring: the sabotage that matters here is appending the
    # directive UNCONDITIONALLY, which leaves the words "if fit_seen:" sitting
    # innocently somewhere else in the handler while the control's prompt
    # silently gains a block. So find the guard branch itself and require the
    # append to live inside it — and nowhere else.
    gate = next((n for n in ast.walk(search_fn)
                 if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "fit_seen"), None)
    inside = "".join(ast.get_source_segment(src, st) or "" for st in gate.body) if gate else ""
    ok("there is a branch on fit_seen", gate is not None)
    ok("the directive is appended INSIDE it", "FIT_DIRECTIVES" in inside)
    # Counted over the AST, not the text: the first version of this line read
    # the source and counted my own comment ("See FIT_DIRECTIVES above") as a
    # second append. A guard that can be satisfied — or broken — by a comment is
    # not reading the code.
    uses = sum(1 for n in ast.walk(search_fn) if isinstance(n, ast.Name) and n.id == "FIT_DIRECTIVES")
    ok("...and nowhere else in the handler", uses == 1, f"{uses} use(s) in code")
    ok("the candidate list is appended unconditionally", "candidates_text" not in inside)
    ok("the directive precedes the candidate list", i_fit is not None and i_cand is not None
       and i_fit < i_cand, f"line {i_fit} then line {i_cand}")
    ok("the cache breakpoint is still on the stable half only",
       body.count('"cache_control"') == 1)
    ok("thinking is still explicitly disabled on this call",
       '"type": "disabled"' in body or "'type': 'disabled'" in body)

# ── 6. the rules it sits beside are intact ───────────────────────────
head("6. the directives it sits beside are intact")
ok("REASON1's unmet rule still there", "put it in `unmet` in their own words" in directives)
ok("CATEGORY still governs picks", "your picks MUST be of that category" in directives)
ok("a typed key still outranks tempo and genre",
   "A KEY THE USER TYPED OUTRANKS TEMPO, GENRE AND THE PROJECT'S KEY" in directives)
ok("plain text only still holds", "PLAIN TEXT ONLY" in directives)

print(
    "\n  FIT2 prompt: the model reads the fit, the control does not, the ask still wins — "
    + ("green" if not failures else f"RED ({failures})")
)
sys.exit(1 if failures else 0)
