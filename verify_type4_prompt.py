"""TYPE4 — the search prompt must forbid what the app cannot know.

Zee's screenshot showed a reply claiming Cratify could not see his
inserts (false once a SONG block exists) and quoting "200-400 Hz"
(invented — plugin parameters are not parsed). The detector catches
these questions first now; this is the belt for when one slips past.

Lifted from the REAL source so a deleted directive fails here.
"""
import ast, sys

SRC = "/Users/zee/Desktop/SORT DROP/SortDrop_Code/cratify-backend/server.py"
tree = ast.parse(open(SRC).read())
directives = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "SEARCH_SYSTEM_DIRECTIVES":
                directives = ast.literal_eval(node.value)
if directives is None:
    print("FAIL  SEARCH_SYSTEM_DIRECTIVES not found")
    sys.exit(1)

fails = 0
def ok(label, cond, detail=""):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        fails = 1

low = directives.lower()
ok("the SONG block is named in the prompt", "song (from the open fl project)" in low)
ok("...and the model is told it CAN see the project",
   "you can see their project" in low)
ok("...and told never to claim otherwise",
   "never say you cannot see it" in low)
ok("frequencies are forbidden outright", "never quote a frequency" in low)
ok("...naming the units, so the rule is unambiguous",
   "hz" in low and "db" in low)
ok("...and the example from the real failure is there", "200-400 hz" in low)
ok("listening judgements are forbidden", "you cannot hear the audio" in low)
ok("...with the words that were used", all(w in low for w in ("muddy", "boomy", "harsh")))
ok("...and the honest alternative is stated",
   "saying what you can see is the honest answer" in low)

print("\n  TYPE4 prompt directives: " + ("RED" if fails else "green"))
sys.exit(fails)
