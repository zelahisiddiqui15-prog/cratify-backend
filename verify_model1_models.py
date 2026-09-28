"""MODEL1 — the model is an env var, never a request field, and always on the list.

Zee's ruling: "Model override: build it server-side only (env/config on Railway),
never chosen by the client. Needed later for an Opus test on the coach."

The reason is the meter rather than taste: `meter_gate` counts calls by KIND, not
by model price, so a client-chosen model would let a caller bill Opus against a
Sonnet allowance.

And the swap itself: Sonnet 4.6 is legacy at $3/$15, Sonnet 5 is $2/$10 — same
tier, 33% cheaper, on 89% of the spend. The one hazard is that Sonnet 5 runs
ADAPTIVE thinking when `thinking` is omitted where 4.6 ran it off, so every call
site must disable it explicitly or the swap quietly buys thinking tokens.
"""
import ast
import os
import re
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
src = open(SRC).read()
failures = 0


def ok(label, cond, detail=""):
    global failures
    if not cond:
        failures += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))


def head(s):
    print(f"\n  {s}:")


head("1. every model comes from the env, with an allowlist")
for name, env in [("SEARCH_MODEL", "CRATIFY_SEARCH_MODEL"), ("COACH_MODEL", "CRATIFY_COACH_MODEL"),
                  ("REF_MODEL", "CRATIFY_REF_MODEL"), ("SMALL_MODEL", "CRATIFY_SMALL_MODEL")]:
    m = re.search(rf"^{name} = _model\(\"{env}\", ", src, re.M)
    ok(f"{name} reads {env} through _model()", m is not None)
# PARSED, not grepped. The first version tested whether the string
# "claude-sonnet-5" appeared ANYWHERE in the file — and it still does, as the
# SEARCH_MODEL default, so deleting it from the allowlist left the check green.
# The invariant that actually matters is the one that lets the server start:
# every default must be a member, or _model() raises at import.
allow = None
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "MODEL_ALLOWLIST" for t in node.targets
    ):
        allow = {ast.literal_eval(e) for e in node.value.elts}
ok("MODEL_ALLOWLIST is a literal set the guard can read", isinstance(allow, set) and len(allow) > 0,
   f"{sorted(allow) if allow else 'not found'}")
defaults = dict(re.findall(r'^(\w+) = _model\("[^"]+", "([^"]+)"\)$', src, re.M))
ok("every hard-coded default is IN the allowlist — otherwise the server cannot start",
   bool(defaults) and all(v in (allow or set()) for v in defaults.values()),
   ", ".join(f"{k}={v}" for k, v in defaults.items()))
ok("...and Sonnet 5 is one of them", "claude-sonnet-5" in (allow or set()))
ok("an unknown model REFUSES TO START rather than failing every call",
   "refusing to start" in src and "raise RuntimeError" in src)

head("2. the client cannot choose the model")
# /classify_batch is the one endpoint that ever read a model from the body; it is
# bounded by the same allowlist. No OTHER endpoint may read one.
body_reads = re.findall(r'data\.get\("model"\)', src)
ok("at most ONE endpoint reads a model from the request body", len(body_reads) <= 1, f"{len(body_reads)} site(s)")
for ep in ["SEARCH_MODEL", "COACH_MODEL", "REF_MODEL"]:
    ok(f"{ep} is used as a constant, never overridden per request",
       f"model={ep}" in src)

head("3. thinking is disabled wherever it can cost — the Sonnet 5 hazard")
# Per CALL, by which model constant it uses. Sonnet 5 runs ADAPTIVE thinking when
# `thinking` is omitted where 4.6 ran it off, so the swap would quietly buy
# thinking tokens on any call that leaves it out.
tree = ast.parse(src)
big, big_off, small, small_off = 0, 0, 0, 0
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    kw = {k.arg for k in node.keywords if k.arg}
    if "model" not in kw or "max_tokens" not in kw:
        continue
    model_src = ""
    for k in node.keywords:
        if k.arg == "model":
            model_src = ast.unparse(k.value)
    has = "thinking" in kw
    if any(n in model_src for n in ("SEARCH_MODEL", "COACH_MODEL", "REF_MODEL")):
        big += 1
        big_off += 1 if has else 0
    else:
        small += 1
        small_off += 1 if has else 0
ok(f"every call on the Sonnet/Opus tier disables thinking ({big} call(s))", big > 0 and big_off == big,
   f"{big_off} of {big}")
# The SMALL_MODEL calls deliberately omit it: adding an untested parameter to five
# working paid endpoints is its own risk, and Haiku 4.5's default is not measured
# here. THE TRAP IS THAT THIS IS ONLY SAFE WHILE SMALL_MODEL IS HAIKU — the
# planned labelling test points it at Sonnet 5, which is exactly when it stops
# being safe. So the condition is enforced rather than remembered.
small_is_haiku = 'SMALL_MODEL = _model("CRATIFY_SMALL_MODEL", "claude-haiku-4-5-20251001")' in src
ok(f"the {small} small-model call(s) omit thinking, and SMALL_MODEL defaults to Haiku",
   small_is_haiku, f"{small - small_off} omit it")
ok("...and the risk is written down where the next person will set that env var",
   "adaptive thinking" in src.lower() and "CRATIFY_SMALL_MODEL" in src)
ok("...and pass it DISABLED, since Sonnet 5 would otherwise run adaptive",
   src.count('thinking={"type": "disabled"}') >= 3, str(src.count('thinking={"type": "disabled"}')))

head("4. /health says which models are serving")
ok("the models are in the payload, so a deploy is verifiable from outside",
   '"models": {' in src and '"search": SEARCH_MODEL' in src)

print("\n  MODEL1 models: server-side only, allowlisted, thinking off, and reported — "
      + ("green" if not failures else f"RED ({failures})"))
sys.exit(1 if failures else 0)
