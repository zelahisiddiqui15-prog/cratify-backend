"""REF1 — _reference_read against every response shape, no network.

server.py cannot be imported here (its deps live on Railway), so the
function is lifted out of the REAL source by AST and executed. That is
deliberately not a copy: if the source changes, this runs the change.
"""
import ast, sys, types

SRC = "/Users/zee/Desktop/SORT DROP/SortDrop_Code/cratify-backend/server.py"
tree = ast.parse(open(SRC).read())
fn = next((n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "_reference_read"), None)
if fn is None:
    print("FAIL  _reference_read is not defined in server.py")
    sys.exit(1)
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), SRC, "exec"), ns)
_reference_read = ns["_reference_read"]

def B(**kw): return types.SimpleNamespace(**kw)
def resp(blocks): return types.SimpleNamespace(content=blocks)

fails = 0
def ok(label, cond, detail=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond: fails = 1

r = resp([
    B(type="web_search_tool_result", content=[B(url="https://songbpm.com/a", title="A"),
                                              B(url="https://songbpm.com/a", title="dup")]),
    B(type="tool_use", name="report_reference",
      input={"found": True, "title": "T", "artist": "A", "bpm": 107, "key": "F minor",
             "source_url": "https://songbpm.com/a"}),
])
f, urls = _reference_read(r)
ok("the tool's input is what comes back", f.get("title") == "T")
ok("result urls are de-duplicated", len(urls) == 1, str(urls))

r = resp([B(type="web_search_tool_result", content=[B(url="https://x/y", title="Y")]),
          B(type="text", text="I could not identify it.")])
f, urls = _reference_read(r)
ok("searched but never called the tool reads as found=false", f.get("found") is False)
ok("...and the urls it did fetch are still reported", len(urls) == 1)

r = resp([B(type="tool_use", name="report_reference",
            input={"found": False, "title": "Guess", "bpm": 120, "key": "C minor",
                   "genre": "house", "artist": "Nobody", "vibe": ["warm"],
                   "source_url": "https://made.up/"})])
f, _ = _reference_read(r)
for field in ("title", "artist", "genre", "bpm", "key", "vibe", "source_url"):
    ok(f"a not-found answer cannot smuggle a {field}", f.get(field) is None, repr(f.get(field)))

r = resp([B(type="tool_use", name="something_else", input={"found": True, "title": "X"})])
f, _ = _reference_read(r)
ok("another tool's input is not read as a reference", f.get("found") is False)

r = resp([B(type="tool_use", name="report_reference", input={"found": "yes", "title": "T"})])
f, _ = _reference_read(r)
ok("found is coerced to a real bool, not trusted", f.get("found") is True, repr(f.get("found")))

r = resp([])
f, urls = _reference_read(r)
ok("an empty response is found=false, not a crash", f.get("found") is False)

print("\n  REF1 backend read: " + ("RED" if fails else "green"))
sys.exit(fails)
