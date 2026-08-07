import os
import json
import anthropic
import stripe
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import voyageai
import re as _re

VOYAGE_CLIENT = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))
EMBED_MODEL = "voyage-4-lite"
EMBED_DIMENSION = 512
from models import (init_db, create_user, get_user, get_user_by_email,
                    get_user_by_username, username_exists, get_db,
                    increment_sorts, activate_subscription,
                    get_usage, add_usage, month_key,
                    deactivate_subscription, set_stripe_customer, hash_password)

load_dotenv()

# --- Gate BE1 --- model ids in ONE place per tier, and the /search
# stable prefix hoisted to module level: it must be a byte-stable prefix
# for prompt caching to have anything to match on.
SMALL_MODEL = "claude-haiku-4-5-20251001"

# ── METER1: monthly ceilings for BACKGROUND work ─────────────────────────
# RECOMMENDED NUMBERS (Zee approves before deploy — see the gate report):
#   EMBED_MONTHLY_LIMIT    30,000/user/month  ≈ $3.00 max COGS (Voyage)
#   CLASSIFY_MONTHLY_LIMIT 10,000/user/month  ≈ $4.50 max COGS (Haiku)
# Rationale: one full index of a 30k-file library per month, with the
# classify share sized to the measured ~1/3 of files that reach tier 5.
# Typical cost is far lower (indexing is one-time); the cap bounds abuse
# and runaway clients, not honest use. Background work does NOT touch the
# trial — the trial gate meters user-initiated actions only.
EMBED_MONTHLY_LIMIT = 30000
CLASSIFY_MONTHLY_LIMIT = 10000
# METER2 ceilings — recurring-use endpoints. The client's $5/week budget
# is the throttle an honest user feels; these are the abuse backstop and
# sit strictly ABOVE honest heavy use so the two limits never disagree
# in practice (server >> client-implied volume):
#   search    2,000/mo  (~$30-40 max; $5/wk client cap implies ~1,400/mo)
#   suggest   3,000/mo  (~$9 max; auto-fires on chat open — S6 leaked twice)
#   describe    500/mo  (~$10 max; one call per attached reference)
#   intent    2,000/mo  (tiny Haiku; legacy caller only)
#   summarize   200/mo  (rare user action; legacy caller only)
SEARCH_MONTHLY_LIMIT = 2000
SUGGEST_MONTHLY_LIMIT = 3000
DESCRIBE_MONTHLY_LIMIT = 500
INTENT_MONTHLY_LIMIT = 2000
SUMMARIZE_MONTHLY_LIMIT = 200


def meter_gate(data, kind, counter_field, limit, count=1):
    """METER2 — the one shape every spending endpoint uses: identity
    required (400/404), ceiling checked BEFORE spend (429 monthly_limit),
    counting done by the caller AFTER success. Returns (error_response,
    status, user_id); error_response is None when the call may proceed."""
    user_id = (data or {}).get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400, None
    if not get_user(user_id):
        return jsonify({"error": "user not found"}), 404, None
    used = get_usage(user_id).get(counter_field, 0) or 0
    if used + count > limit:
        return jsonify({"error": "monthly_limit", "kind": kind,
                        "used": used, "limit": limit,
                        "month": month_key()}), 429, None
    return None, 200, user_id

# AI-OPT: models the batch endpoint may be asked to use — an allow-list so
# the API can never be pointed at an expensive model by a caller.
ALLOWED_CLASSIFY_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
}

# Shared by /classify and /classify_batch — one source of truth so the two
# endpoints can never drift apart in vocabulary (the singular/plural traps
# were born from exactly that kind of duplication).
CLASSIFY_RULES = """You are a music file classifier for a producer tool called Cratify.

Analyze each filename and return a JSON object with these fields:
- category: EXACTLY one of: Bass, Chord, Drums, FX, Guitar, Keys, Melody, Pad, Synth, Vocals, Ambient, Other. These exact spellings, including the plurals Drums and Vocals — never Drum, Vocal, Piano, Lead, Pluck, Arp, Strings, Brass, or Texture.
- instrument: EXACTLY one of (these spellings, capitalized): 808, Acapella, Adlib, Ambient, Arp, Atmosphere, Bass, Brass, Cello, Chop, Chord, Clap, Clarinet, Cymbal, Downlifter, Drums, FX, Flute, Guitar, Guitar Acoustic, Guitar Electric, Harp, Hat, Horn, Impact, Keys, Kick, Lead, Loop, Melody, Oboe, Organ, Other, Pad, Percussion, Piano, Pluck, Reese, Rhodes, Rim, Riser, Saw, Sax, Shaker, Snare, Strings, Sub, Sweep, Synth, Texture, Tom, Transition, Trombone, Trumpet, Viola, Violin, Vocals, Wurli. Null if none fits.
- drum_type: ONLY for Drums, EXACTLY one of: Kick, Snare, Hat, Clap, Percussion, Cymbal, Tom, Rim, Shaker. These spellings — never Hi-Hat, Perc, or Full Loop. Null otherwise.
- subcategory: more specific description
- key: musical key if detectable (e.g. "Am", "C#") or null. Always null for drums.
- bpm: BPM ONLY if a number is visible in the filename, else null. Never guess a tempo.
- file_type: "stem", "preset", "midi", "sample", or "loop"
- confidence: 0 to 1

CRITICAL CATEGORY RULES:
Where the instrument suggests a category outside the list, map it:
- piano, organ, rhodes, keyboard → Keys
- strings, brass, orchestral, sax, flute, harp, woodwind → Melody
- lead, pluck, arp → Synth
- texture, drone → Ambient
- any drum or percussion → Drums (plural, always)
- any vocal, vox, acapella → Vocals (plural, always)

Never use Loop as a standalone category. Classify a loop by its instrument:
- Drum Loop, Beat, Break → category: Drums (drum_type: null — it is a whole loop, not one drum)
- Bass Loop → category: Bass
- Synth Loop, Synth Riff, Lead Loop, Arp Loop → category: Synth
- Piano Loop → category: Keys
- Guitar Loop → category: Guitar
- Chord Loop, Chord Stab → category: Chord
- Melody Loop → category: Melody
- Vocal Loop, Vox Loop → category: Vocals
- Pad Loop, Atmosphere Loop → category: Pad
- FX Loop, Riser, Sweep → category: FX
- Texture Loop, Ambient Loop → category: Ambient
- If a loop's instrument is truly unknown: category: Other

Set file_type to "loop" whenever the filename contains "loop", "lp", "riff", "break", or "beat".

"""

FALLBACK_RESULT = {
    "category": "Other", "instrument": None, "drum_type": None,
    "subcategory": "Unknown", "key": None, "bpm": None,
    "file_type": "sample", "confidence": 0.3,
}


def postprocess_result(result, filename):
    """Per-file post-processing shared by both classify endpoints."""
    if result.get("category") in ("Drum", "Drums"):
        result["key"] = None
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("fxp", "vital", "nmsv", "xpf", "aupreset", "patch"):
        # AI-OPT fix: file_type is forced, the CATEGORY the model chose
        # stands. The old code forced category='Preset', which is not in
        # the 13-canon — the client guard coerced it to 'Other' and the
        # model's real answer was destroyed for every preset file.
        result["file_type"] = "preset"
        result["key"] = None
        result["bpm"] = None
    return result
# PRICING WATCH, dated 2026-07-28 (Gate BE1). Staying on claude-sonnet-4-6
# is a MEASURED choice, not inertia. claude-sonnet-5 is $2/$10 per MTok on
# introductory pricing through 2026-08-31, then $3/$15 — the same sticker as
# 4.6, but NOT the same cost: its tokenizer bills ~22% more tokens for
# identical text (measured on this very prompt: the /search stable prefix is
# 1,995 tokens on 4.6 vs 2,436 on 5). So from 2026-09-01 an identical
# request costs ~22% MORE on Sonnet 5, permanently.
# Also note: Sonnet 5 runs ADAPTIVE thinking when `thinking` is omitted,
# unlike 4.6 — see the explicit {"type": "disabled"} at the /search call.
SEARCH_MODEL = "claude-sonnet-4-6"

SEARCH_SYSTEM_DIRECTIVES = """You are Cratify, an AI music producer assistant. You help producers find the perfect sample from their personal library.

The user will ask for sounds. You have been given the top-50 most semantically similar samples from their library, pre-ranked by vector similarity. Your job is to analyze them, identify the best matches, write a concise producer-friendly explanation, and infer the broader filter criteria for a "see more" button.

Guidelines for your response:
- PERSONALIZATION: the query may begin with a "USER PREFERENCES" block and/or a "RECENT CORRECTIONS IN THIS CHAT" block. Treat every preference line as a standing instruction — check your response against each before finalizing. Preferences shape tone, pick ORDERING (e.g. "show Orbit samples first when they match" = lead with matching Orbit-pack files where they genuinely fit), and explanation style. They are best-effort ranking hints, NOT hard filters, and they NEVER override truthfulness — a preference cannot invent a match or reorder in a sound that does not fit. Honor RECENT CORRECTIONS literally: do not repeat a mistake the user just thumbed down.
- CONTEXT: the conversation history precedes the latest message. If the latest message is a follow-up ("okay how about vital?", "something darker", "more like that", "in F minor instead"), interpret it AS A REFINEMENT of the previous request in this thread — never as a cold literal search. "okay how about vital?" after a bass hunt means "that same bass search, but Vital presets." A follow-up inherits the instrument/vibe/context of what came before unless it clearly changes them.
- CATEGORY: if the user names an instrument or category (drums, bass, pads, vocals, chords, keys, leads...), your picks MUST be of that category. A follow-up that names NO category inherits the category of the previous turn — "okay how about serum?" after a bass hunt means SERUM BASS presets, not whatever else Serum makes — unless the user names a new one, which replaces it. The candidate list is already weighted toward it. If you include an off-category pick, your reply MUST say why it earns its place ("threw in a pad since it doubles as a bass layer"). Otherwise, stay on-category.
- picks: identify the 4-8 best actual matches. If fewer than 4 truly match, return fewer. If nothing matches well, return empty list.
- reply: talk like a producer friend texting back — plain, warm, concrete. Roughly 2-3 short sentences. One short paragraph is usually right; two at most.
  A CRATE, NOT AN ARRANGEMENT. This is the governing rule for the reply, and it overrides any older "lead with the move" instinct. Producers audition many options before choosing one. THE RESULTS PANEL IS THE ANSWER — your reply only orients them to it. Naming three files and moving into a layering plan does the choosing for them, and choosing is the producer's job, not yours. Cratify is a producer's helper, not a producer.
  ANSWER ONLY WHAT WAS ASKED. Asked for chords, talk about chords. Do NOT volunteer bass, drums, leads, percussion, structure, or a layering plan in the same reply — not as a bonus, not as a closing suggestion, not as "you could also". If they want the next piece they will ask for it, and the suggestion chips are already there to offer it.
  So: say what is in the crate and why these fit — key, tempo, character — enough to start auditioning. Do NOT reduce it to a shortlist of two or three "best" files and an arrangement around them; that pre-decides the very thing they opened the app to decide.
  PROMOTE AUDITIONING where it reads naturally: "worth hearing a few before you commit", "flick through these and see which one sits". That is the behaviour to encourage.
  NEXT STEPS BELONG IN THE CHIPS, NOT THE PROSE. The UI already renders follow-up suggestions; that is where "now find a bassline" lives. Never put it in the reply.
  ASSUME NO THEORY BACKGROUND. A bedroom producer who has never heard "modal", "diatonic", "tonality" or "relative minor" must be able to act on every sentence you write. Say "sounds darker" not "minor tonality"; "sits lower under everything" not "an octave below the root"; "these two sit together" not "they share a key centre". If you name a key or BPM, attach the action to it — but read the WARP ENGINE block first (see below), because when the app is already matching key/tempo the action is "it just fits", not "nudge it up".
  Use a theory term ONLY if the user's own message used that kind of language first. Then match their level and go as deep as they did — a user who asks about modal interchange gets a real answer, not a dumbed-down one.
  Do NOT wrap filenames or file references in ** or any markdown emphasis. The UI turns file references into interactive chips, and stray asterisks render as literal characters.
  Short sentences. No run-ons. No lecturing.
- INTERVALS: state them in SEMITONES, or do not name them at all. Write "up 7 semitones" or "3 semitones down" — never "a perfect fourth", "a minor third", "a fifth". Interval names are easy to get wrong (C up to G is a FIFTH, not a fourth) and a producer who does know theory stops trusting the whole reply after one slip. The semitone count is unambiguous and is the number they type into a pitch knob anyway. If the user's own message used an interval name you may mirror it, but still give the semitone count alongside.
- WARP ENGINE (FIX5): the context may carry a "WARP ENGINE:" line saying whether Auto-BPM and Auto-pitch are ON. This app OWNS a time-stretch/transpose engine, and that line is the only place you can learn whether it is running. Obey it literally.
  When an axis is ON, files from the user's library are ALREADY matched on that axis the moment they audition or drag them. Do NOT tell the user to transpose, repitch, pitch up/down, timestretch, or change the tempo of a library file on an axis that is ON. They would be doing work the app already did, and applying it twice lands them further from their project than doing nothing. Say it lands in key/tempo automatically, then spend your sentences on the actual musical choice.
  This does NOT make you vague about key or BPM — keep naming them. "It's in C, 8 semitones off, but it'll land in your key on its own" is right; "it's in C, so pitch it up 8 semitones" is wrong when Auto-pitch is ON.
  The WARP LIMITS line names where the engine genuinely cannot help — MIDI (no audio to warp), one-shots (never pitch-shifted), files past the stretch clamp, files with no detected key/BPM. For THOSE, manual instructions are correct and you should still give them, plainly.
  When the block says both are OFF, or there is no WARP ENGINE line at all, nothing is automatic — give normal manual tempo/pitch instructions.
- RECENTLY TRIED (FIX7): the context may carry a "RECENTLY TRIED IN THIS PROJECT" line listing files the user dragged out of Cratify recently. Read it as EVIDENCE OF DIRECTION AND TASTE — what they are reaching for right now, in this song.
  It is NOT a record of usage. A drag means the user pulled the file in to try it; they may have kept it or deleted it thirty seconds later inside their DAW, and Cratify cannot see which. So NEVER say or imply that one of these is "in your track", "already in the song", "the kick you're using", or that they added/chose/committed to it. If you refer to one at all, the honest framing is "the one you pulled in earlier" or "since you were leaning toward X".
  Practical use: do not re-recommend a file that is already on that list unless you have a specific reason, and say the reason ("still the best fit for this, even though you've already tried it"). Prefer offering things that COMPLEMENT the direction the list shows.
  If there is no such line, the user has tried nothing in this project yet — say nothing about it either way.
- NEVER mention the [ID] numbers in your reply text.
- HOW TO NAME A FILE IN PROSE: use a SHORT HUMAN DESCRIPTOR, not the raw filename. Write "the smooth 808", "that reso 808", "the KSHMR sub" — NOT "91V_LRB_808_smooth_C.wav". Long underscored filenames are unreadable mid-sentence and the UI renders your descriptor as a clickable chip anyway, so the user never needs to see the raw name to act on it.
- mentions (REQUIRED FIELD): for EVERY file reference in your reply, add {id, text} where `text` is the EXACT substring of your reply naming it, copied character-for-character — same spelling, same case, no trailing punctuation — so it can be found by plain string search. If your reply mentions three files, mentions has three entries. This is what turns your words into play/reveal buttons; a reference with no mentions entry is dead text to the user. Only ids from the candidate list. If the reply genuinely names no file, return an empty array.
  EACH `text` SPAN MUST BE UNIQUE WITHIN THE REPLY. The span is found by plain string search, so two mentions sharing the same wording are indistinguishable — one file silently stands in for both and the user cannot tell which one they are hearing. If you want to point at two files that would naturally share a phrase ("the Au5 dub kicks"), either name them distinctly ("the first Au5 dub kick" / "the second Au5 dub kick") or refer to only one of them. Never emit two mentions with identical `text`.
- filters_used: describes the broader search the user might want ("all vocal chops in Gm around 140 BPM") - be permissive, it's an escape hatch. Any field can be omitted if not inferrable.
- category MUST be one of the library's real category values: Ambient, Bass, Chord, Drums, FX, Guitar, Keys, Melody, Other, Pad, Synth, Vocals. These are the values files actually carry — any other word makes the 'see more' filter match nothing.
- progression: ONLY when your reply prescribes a chord sequence (e.g. "Fm -> Db -> Ab -> Eb, i -> VI -> III -> VII") AND candidate files match its chords (many filenames carry roman numerals and chord names — e.g. "i - F min.mid", "VI - Db Maj.mid"): populate progression with one entry per step IN PLAYING ORDER, mapping each step to the candidate ids whose filename matches that chord, best first. A step with no matching file gets an empty pick_ids — NEVER force a bad match. Your reply prose is unchanged either way.

You MUST respond by calling the return_search_results tool. Do not respond with text.

The user's library samples (pre-ranked by similarity, ID in brackets):
"""

SEARCH_TOOL_SCHEMA = {
    "name": "return_search_results",
    "description": "Return ranked sample picks with an explanation and inferred filters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "picks": {
                "type": "array",
                "description": "Best matching samples, 0-8 items, ranked by relevance.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Sample ID from the candidates list."},
                        "score": {"type": "number", "description": "Match quality 0.0-1.0."},
                        "reason": {"type": "string", "description": "Short phrase explaining why this matches."},
                    },
                    "required": ["id", "score", "reason"],
                },
            },
            "reply": {
                "type": "string",
                "description": "Producer-friendly explanation, 2-3 short paragraphs separated by blank lines.",
            },
            "progression": {
                "type": "array",
                "description": "OPTIONAL — only when the reply prescribes a chord sequence. One entry per step, playing order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "integer", "description": "1-based position in the sequence."},
                        "roman": {"type": "string", "description": "Roman numeral, e.g. 'i', 'VI'."},
                        "chord": {"type": "string", "description": "Chord name, e.g. 'Fm', 'Db maj'."},
                        "pick_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Candidate ids whose filename matches this chord, best first. Empty if none match — never force.",
                        },
                    },
                    "required": ["step", "roman", "chord", "pick_ids"],
                },
            },
            "mentions": {
                "type": "array",
                "description": "REQUIRED. One entry per place the reply text refers to a specific candidate file. Empty array [] if the reply names no file at all.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Candidate id being referred to."},
                        "text": {
                            "type": "string",
                            "description": "The EXACT substring of the reply that names this file, copied character-for-character.",
                        },
                    },
                    "required": ["id", "text"],
                },
            },
            "filters_used": {
                "type": "object",
                "description": "Broader filter criteria the user might want (for 'see more' button).",
                "properties": {
                    # CANON1 follow-up: enum enforces the canon the DB
                    # actually contains — the old free string let the model
                    # emit 'Leads'/'Loops'/'One-Shots', values no files row
                    # has ever held, so the see-more filter matched nothing.
                    "category": {"type": "string", "enum": [
                        "Ambient", "Bass", "Chord", "Drums", "FX", "Guitar",
                        "Keys", "Melody", "Other", "Pad", "Synth", "Vocals",
                    ]},
                    "key": {"type": "string"},
                    "bpm_min": {"type": "integer"},
                    "bpm_max": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "required": ["picks", "reply", "filters_used", "mentions"],
    },
}


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 204

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Cratify API"})


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.json or {}
    email = data.get("email")
    password = data.get("password")
    username = data.get("username")

    if not email:
        return jsonify({"error": "email required"}), 400

    existing = get_user_by_email(email)
    if existing:
        return jsonify({"error": "email already registered"}), 409

    if username and username_exists(username):
        return jsonify({"error": "username already taken"}), 409

    user_id = create_user(email=email, username=username, password=password)
    return jsonify({
        "user_id": user_id,
        "username": username,
        "sorts_remaining": 25,
        "subscription_active": False
    })


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.json or {}
    identifier = data.get("email") or data.get("identifier")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"error": "email/username and password required"}), 400

    user = get_user_by_email(identifier)
    if not user:
        user = get_user_by_username(identifier)

    if not user:
        return jsonify({"error": "invalid credentials"}), 401

    if user.get("password_hash") != hash_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])
    return jsonify({
        "user_id": user["id"],
        "username": user.get("username"),
        "email": user["email"],
        "sorts_remaining": sorts_remaining if not user["subscription_active"] else None,
        "subscription_active": bool(user["subscription_active"])
    })


@app.route("/auth/check-username", methods=["GET"])
def check_username():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "username required"}), 400
    taken = username_exists(username)
    return jsonify({"available": not taken, "username": username})


@app.route("/subscription/status", methods=["GET"])
def subscription_status():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    sorts_remaining = None
    if not user["subscription_active"]:
        sorts_remaining = max(0, user["trial_limit"] - user["sorts_used"])

    return jsonify({
        "subscription_active": bool(user["subscription_active"]),
        "sorts_used": user["sorts_used"],
        "sorts_remaining": sorts_remaining,
        "trial_limit": user["trial_limit"],
        "username": user.get("username"),
        "email": user.get("email")
    })


@app.route("/classify", methods=["POST"])
def classify():
    data = request.json or {}
    filename = data.get("filename")
    user_id = data.get("user_id")

    if not filename or not user_id:
        return jsonify({"error": "filename and user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    # METER1 — background work is exempt from the trial (the gate was
    # written for user-initiated actions) but answers to a monthly ceiling.
    source = data.get("source", "interactive")
    if source == "background":
        used = get_usage(user_id)["classify_bg_count"]
        if used + 1 > CLASSIFY_MONTHLY_LIMIT:
            return jsonify({"error": "monthly_limit", "kind": "classify",
                            "used": used, "limit": CLASSIFY_MONTHLY_LIMIT,
                            "month": month_key()}), 429
    elif not user["subscription_active"]:
        if user["sorts_used"] >= user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    prompt = CLASSIFY_RULES + f"""Filename: {filename}

Return ONLY valid JSON. No markdown, no explanation."""


    try:
        message = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
    except Exception as api_err:
        print(f"[classify] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": str(api_err)}), 500

    try:
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
    except Exception:
        result = {
            "category": "Other", "drum_type": None, "subcategory": "Unknown",
            "key": None, "bpm": None, "file_type": "stem", "confidence": 0.5
        }

    result = postprocess_result(result, filename)
    if source == "background":
        add_usage(user_id, classify_bg=1)
    else:
        increment_sorts(user_id)
        add_usage(user_id, classify_int=1)
    return jsonify(result)


@app.route("/classify_batch", methods=["POST"])
def classify_batch():
    """AI-OPT — N filenames per call, one metered sort each.

    The ~700-token rule prompt is paid once per CALL, so a batch of 25
    amortises the expensive part ~25x. Fail-soft contract: a malformed
    ITEM becomes the honest low-confidence fallback for that item only; a
    malformed WHOLE RESPONSE is a 502 and the client falls back to
    singles for that batch — one bad answer never costs the other 24.
    """
    data = request.json or {}
    filenames = data.get("filenames")
    user_id = data.get("user_id")
    if not isinstance(filenames, list) or not filenames or not user_id:
        return jsonify({"error": "filenames (list) and user_id required"}), 400
    if len(filenames) > 50:
        return jsonify({"error": "max 50 filenames per call"}), 400
    if not all(isinstance(f, str) and f for f in filenames):
        return jsonify({"error": "filenames must be non-empty strings"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404
    source = data.get("source", "interactive")
    if source == "background":
        used = get_usage(user_id)["classify_bg_count"]
        if used + len(filenames) > CLASSIFY_MONTHLY_LIMIT:
            return jsonify({"error": "monthly_limit", "kind": "classify",
                            "used": used, "limit": CLASSIFY_MONTHLY_LIMIT,
                            "month": month_key()}), 429
    elif not user["subscription_active"]:
        if user["sorts_used"] + len(filenames) > user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    model = data.get("model") or SMALL_MODEL
    if model not in ALLOWED_CLASSIFY_MODELS:
        return jsonify({"error": "model not allowed"}), 400

    listing = "\n".join(f"{i + 1}. {fn}" for i, fn in enumerate(filenames))
    prompt = CLASSIFY_RULES + (
        f"You will be given {len(filenames)} filenames. Return ONLY a JSON "
        f"array of exactly {len(filenames)} objects, one per filename IN "
        f"ORDER, each with the fields above.\n\nFilenames:\n{listing}\n\n"
        "Return ONLY the JSON array. No markdown, no explanation."
    )

    try:
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=128 + 90 * len(filenames),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as api_err:
        print(f"[classify_batch] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": str(api_err)}), 500

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        parsed = json.loads(raw.strip())
        if not isinstance(parsed, list) or len(parsed) != len(filenames):
            raise ValueError(f"expected {len(filenames)} items, got "
                             f"{len(parsed) if isinstance(parsed, list) else type(parsed)}")
    except Exception as parse_err:
        # Whole-response failure: bill nothing, client retries as singles.
        print(f"[classify_batch] parse failure: {parse_err}", flush=True)
        return jsonify({"error": "batch_parse_failed"}), 502

    results = []
    for fn, item in zip(filenames, parsed):
        if not isinstance(item, dict) or not item.get("category"):
            item = dict(FALLBACK_RESULT)   # per-item fail-soft
        results.append(postprocess_result(item, fn))

    if source == "background":
        add_usage(user_id, classify_bg=len(filenames))
    else:
        increment_sorts(user_id, len(filenames))
        add_usage(user_id, classify_int=len(filenames))
    return jsonify({
        "results": results,
        "model": model,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    })


@app.route("/classify_preset", methods=["POST"])
def classify_preset():
    """Preset-aware classifier — sub-arc 1d.

    Called by PresetExtractor as Tier 4 when Tiers 1-3 (parser, folder,
    filename) leave fewer than 4 tags. Returns granular preset_category,
    vibe_tags, and use_case via tool_use for structured output.

    Auth/billing mirrors /classify: requires user_id, increments sorts,
    returns 402 trial_exhausted for over-quota free users.
    """
    data = request.json or {}
    filename = data.get("filename")
    user_id = data.get("user_id")
    plugin_name = (data.get("plugin_name") or "Unknown").strip() or "Unknown"
    vendor = (data.get("vendor") or "Unknown").strip() or "Unknown"
    folder_name = (data.get("folder_name") or "").strip()
    current_tags = (data.get("current_tags") or "").strip()

    if not filename or not user_id:
        return jsonify({"error": "filename and user_id required"}), 400

    user = get_user(user_id)
    if not user:
        return jsonify({"error": "user not found"}), 404

    if not user["subscription_active"]:
        if user["sorts_used"] >= user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    system_prompt = (
        "You classify music plugin presets (Vital, Serum, Omnisphere, "
        "u-he, FL Studio, etc.) for a producer search tool called Cratify.\n\n"
        "Given a preset filename plus whatever context is available, "
        "determine the granular category, mood/character descriptors, and "
        "intended use case.\n\n"
        "CRITICAL RULES:\n"
        "- preset_category MUST be one of the allowed values in the tool schema.\n"
        "- vibe_tags describe how the preset SOUNDS, not what it IS. "
        "Avoid restating the category as a vibe.\n"
        "- NEVER infer BPM or musical key — presets don't have those.\n"
        "- If filename is generic (e.g., 'Preset 01', 'Init', 'Default') "
        "and there's no other context, return low confidence with best guess.\n"
        "- Plugin/vendor/folder context is signal — use it. A preset under "
        "/Cymatics/Future Bass/ leans dark and evolving even if filename is bland.\n"
        "- Return via the return_preset_classification tool. No prose, no JSON in text."
    )

    user_prompt = (
        f"Filename: {filename}\n"
        f"Plugin: {plugin_name}\n"
        f"Vendor: {vendor}\n"
        f"Parent folder: {folder_name or '(unknown)'}\n"
        f"Existing tags from parser/filename: {current_tags or '(none)'}"
    )

    try:
        response = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[_PRESET_CLASSIFY_TOOL_SCHEMA],
            tool_choice={
                "type": "tool",
                "name": "return_preset_classification",
            },
        )
    except Exception as api_err:
        print(f"[classify_preset] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": str(api_err)}), 500

    # Extract the tool_use block — tool_choice forces this
    tool_use = None
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and \
                getattr(block, "name", None) == "return_preset_classification":
            tool_use = block
            break

    if tool_use is None:
        print(f"[classify_preset] no tool_use returned. Content: "
              f"{response.content}", flush=True)
        return jsonify({"error": "no_tool_use"}), 500

    parsed = tool_use.input  # schema-validated dict

    # Normalize: preset_category to lowercase, vibe_tags to clean list
    preset_category = (parsed.get("preset_category") or "other").lower().strip()
    vibe_tags = [
        str(t).strip().lower()
        for t in (parsed.get("vibe_tags") or [])
        if t and str(t).strip()
    ]
    use_case = (parsed.get("use_case") or "").strip()
    confidence = float(parsed.get("confidence") or 0.5)

    increment_sorts(user_id)
    add_usage(user_id, classify_int=1)

    return jsonify({
        "preset_category": preset_category,
        "vibe_tags": vibe_tags,
        "use_case": use_case,
        "confidence": confidence,
    })


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["client_reference_id"]
        customer_id = session["customer"]
        subscription_id = session["subscription"]
        if user_id:
            set_stripe_customer(user_id, customer_id)
            activate_subscription(customer_id, subscription_id)
    elif event["type"] == "customer.subscription.deleted":
        sub = event["data"]["object"]
        deactivate_subscription(sub["customer"])

    return jsonify({"status": "ok"})


@app.route("/stripe/create-checkout-session", methods=["POST"])
def create_checkout_session():
    data = request.json or {}
    user_id = data.get("user_id")
    try:
        session = stripe.checkout.Session.create(
            ui_mode="embedded_page",
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID"), "quantity": 1}],
            mode="subscription",
            return_url=f"https://www.cratify.app/dashboard?session_id={{CHECKOUT_SESSION_ID}}",
            client_reference_id=user_id,
        )
        return jsonify({"clientSecret": session.client_secret})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Preset classification tool schema (sub-arc 1d) ─────────────────
_PRESET_CLASSIFY_TOOL_SCHEMA = {
    "name": "return_preset_classification",
    "description": (
        "Classify a music plugin preset by granular category and vibe. "
        "Returns structured fields the Cratify desktop app stores in "
        "the files table."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "preset_category": {
                "type": "string",
                "description": (
                    "Granular preset type. MUST be one of: "
                    "bass, lead, pad, pluck, stab, chord, arp, fx, "
                    "keys, perc, drum, vocal, brass, strings, guitar, other."
                ),
            },
            "vibe_tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "minLength": 2,
                    "description": (
                        "A single vibe word like 'dark' or 'warm'. "
                        "Must be a complete word, never a single character "
                        "or comma-separated list."
                    ),
                },
                "description": (
                    "Array of 1-4 short character/mood descriptors. "
                    "Each array element is ONE complete word. "
                    "Correct examples: ['dark', 'evolving'] or "
                    "['warm', 'punchy', 'lofi']. "
                    "INCORRECT (do not do this): "
                    "['d','a','r','k'] or ['dark, evolving']. "
                    "Vocabulary: dark, bright, warm, evolving, punchy, "
                    "dreamy, aggressive, mellow, dirty, clean, vintage, "
                    "modern, lofi, ethereal, huge, tight, wet, dry."
                ),
            },
            "use_case": {
                "type": "string",
                "description": (
                    "Suggested musical context: 'intro', 'drop', "
                    "'breakdown', 'lead line', 'ambient bed', 'chord stab', "
                    "etc. Empty string if unclear."
                ),
            },
            "confidence": {
                "type": "number",
                "description": "0.0-1.0 confidence in the classification.",
            },
        },
        "required": ["preset_category", "vibe_tags", "confidence"],
    },
}


_INTENT_SYSTEM = """You parse music producer chat messages into structured bulk file actions.

Return ONLY a valid JSON object — no explanation, no markdown fences.

Supported actions: "export" (copy files to a destination folder) or "move" (move files).
Return action: null if the message is just a search or question, not a bulk operation.

Output schema:
{
  "action": "export" | "move" | null,
  "filter": {
    "key":      string | null,   // musical key, e.g. "C# minor", "Am", "G major"
    "category": string | null,   // e.g. "loop", "bass", "drum", "vocal", "pad"
    "bpm_min":  number | null,
    "bpm_max":  number | null,
    "file_type": string | null   // extension without dot: "wav", "mp3", "midi"
  },
  "destination": string | null   // absolute path the user mentioned, or null
}

Decision rules:
- "export / send / copy … to …"  → action: "export"
- "move … to …"                  → action: "move"
- "show / find / search / what"  → action: null
- Vague questions with no clear destination → action: null
- If destination folder is not explicitly stated → destination: null

Examples:
  "export all my C# minor loops to /Users/zee/Desktop"
  → {"action":"export","filter":{"key":"C# minor","category":"loop","bpm_min":null,"bpm_max":null,"file_type":null},"destination":"/Users/zee/Desktop"}

  "move all bass wav files to /Users/zee/Music/Project"
  → {"action":"move","filter":{"key":null,"category":"bass","bpm_min":null,"bpm_max":null,"file_type":"wav"},"destination":"/Users/zee/Music/Project"}

  "send everything between 120 and 130 bpm to my desktop"
  → {"action":"export","filter":{"key":null,"category":null,"bpm_min":120,"bpm_max":130,"file_type":null},"destination":"/Users/zee/Desktop"}

  "show me all loops in Am"
  → {"action":null,"filter":{},"destination":null}

  "find dark pads in C minor"
  → {"action":null,"filter":{},"destination":null}
"""


@app.route("/suggest_prompts", methods=["POST"])
def suggest_prompts():
    """Gate C10 — conversation-aware prompt suggestions (endpoint #2 of
    the R1v2 pattern). Input: the project context block (same builder as
    /search feeds), the chat's recent messages, and a compact library
    sketch. Output: 3-4 short producer-voice prompt strings the user
    could send NEXT — actionable for THIS conversation and THIS song,
    never generic, never repeating what was just asked."""
    data = request.json or {}
    # METER2 — auto-fires on chat open; this endpoint has leaked spend
    # twice (S6). Identity + ceiling, counted per successful model call.
    err, code, _sg_uid = meter_gate(data, "suggest", "suggest_count", SUGGEST_MONTHLY_LIMIT)
    if err is not None:
        return err, code
    context_block = data.get("context_block") or ""
    messages = data.get("messages") or []
    sketch = data.get("library_sketch") or ""
    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400
    if not context_block and not messages:
        return jsonify({"error": "context_block or messages required"}), 400

    convo = "\n".join(
        f"{m.get('role','?')}: {str(m.get('content',''))[:300]}"
        for m in messages[-6:] if isinstance(m, dict)
    )
    ctx_part = ("PROJECT CONTEXT:\n" + context_block[:1200]) if context_block else "No project context — suggest library discovery prompts."
    convo_part = ("RECENT CONVERSATION:\n" + convo) if convo else "This is a brand-new empty chat."
    sketch_part = ("LIBRARY SKETCH: " + sketch[:400]) if sketch else ""
    prompt = f"""You suggest the user's NEXT search prompt in Cratify, a sample-library
tool for music producers. The user types prompts to find samples/presets/MIDI.

{ctx_part}

{convo_part}

{sketch_part}

If a "USER PREFERENCES" block is present, honor each line — they shape
tone and phrasing of the suggestions (best-effort, never invent).

Return a JSON array of 3-4 short prompt strings (each under 12 words,
producer voice, first person, concrete). Rules:
- Each must be an actionable NEXT step for THIS conversation and song.
- Plain bedroom-producer language: name the SOUND you'd look for
  ("punchy 140 kick", "warm Rhodes chords"), not music-theory terms.
  Assume NO theory background — every suggestion must be typeable by
  someone who has never heard the words "modal" or "relative minor".
  Use a theory term ONLY if the user's own messages used it first, and
  then match their level.
- Never generic filler; never repeat or trivially rephrase what was just asked.
- If conversation exists, at least 2 suggestions must build on it.
Return ONLY the JSON array, no markdown, no explanation."""

    try:
        message = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        import json as _json
        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").lstrip("json").strip()
        prompts = _json.loads(raw)
        if not isinstance(prompts, list):
            raise ValueError("not a list")
        prompts = [str(x).strip() for x in prompts if str(x).strip()][:4]
        if not prompts:
            raise ValueError("empty")
        add_usage(_sg_uid, suggest=1)
        return jsonify({"prompts": prompts})
    except Exception as api_err:
        print(f"[suggest_prompts] error: {api_err}", flush=True)
        return jsonify({"error": "model_error"}), 502


@app.route("/describe_reference", methods=["POST"])
def describe_reference():
    """Gate R1v2 — producer-voice read of a reference track's measured
    feature timeline. TEXT in, short read out. The client shows it in an
    approve/edit gate (genre calls will sometimes be wrong — the user's
    edit is the correction layer), and the read is CHARACTER context
    only: the project's own key/BPM govern matching client-side."""
    data = request.json or {}
    err, code, _dr_uid = meter_gate(data, "describe", "describe_count", DESCRIBE_MONTHLY_LIMIT)
    if err is not None:
        return err, code
    features_text = data.get("features_text")
    if not features_text or not isinstance(features_text, str):
        return jsonify({"error": "features_text required"}), 400

    prompt = f"""You are an experienced music producer describing a reference track
to a collaborator, based ONLY on measured features (no audio access).

Measured feature timeline:
{features_text[:2000]}

Write 1-3 sentences in a natural producer voice describing the track's
character and structure (energy arc, sections, tone). You may suggest a
genre feel ONLY if the features support it, hedged ("feels like...").
Never invent specifics the features don't show. Return ONLY the
description text, no preamble."""

    try:
        message = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        read = message.content[0].text.strip()
        add_usage(_dr_uid, describe=1)
        return jsonify({"read": read})
    except Exception as api_err:
        print(f"[describe_reference] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": "model_error"}), 502


@app.route("/intent", methods=["POST"])
def intent():
    """Parse a chat message for bulk file action intent using Claude Haiku."""
    data = request.json or {}
    err, code, _in_uid = meter_gate(data, "intent", "intent_count", INTENT_MONTHLY_LIMIT)
    if err is not None:
        return err, code
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"action": None})
    try:
        resp = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=256,
            system=_INTENT_SYSTEM,
            messages=[{"role": "user", "content": message}]
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if model wraps anyway
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rsplit("```", 1)[0]
        result = json.loads(raw.strip())
        # Normalise: always return the top-level action key
        if "action" not in result:
            result["action"] = None
        add_usage(_in_uid, intent=1)
        return jsonify(result)
    except Exception as e:
        print(f"[/intent] error: {e}", flush=True)
        return jsonify({"action": None})


_PAIR_MAP = {
    "kick":      ["snare", "clap", "hi-hat", "hihat"],
    "bass":      ["pad", "lead", "pluck"],
    "lead":      ["pad", "arp"],
    "loop":      ["drum loop", "bass"],
}

@app.route("/pair", methods=["POST"])
def pair():
    """Given a filepath + category, return top 5 complementary files from target categories."""
    data = request.json or {}
    filepath = data.get("filepath", "").strip()
    category = (data.get("category") or "").strip().lower()

    if not filepath or not category:
        return jsonify({"error": "filepath and category required"}), 400

    # Determine target categories from pairing map
    target_cats = None
    for key, targets in _PAIR_MAP.items():
        if key in category:
            target_cats = targets
            break
    if not target_cats:
        return jsonify({"error": f"no pairing defined for category '{category}'"}), 400

    try:
        import sqlite3 as _sqlite3
        import numpy as _np
        import base64 as _b64
        from pathlib import Path as _Path
        from scipy.spatial.distance import cosine as _cosine

        db_path = str(_Path.home() / ".cratify" / "index.db")

        # Load embedding for the query file
        conn = _sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT embedding FROM files WHERE filepath = ?", (filepath,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0]:
            return jsonify({"error": "no embedding found for this file — run indexer first"}), 404

        query_emb = _np.frombuffer(row[0], dtype=_np.float32)

        # Build SQL LIKE clause for target categories (case-insensitive)
        placeholders = " OR ".join(["LOWER(category) LIKE ?" for _ in target_cats])

        conn2 = _sqlite3.connect(db_path)
        c2 = conn2.cursor()
        c2.execute(
            f"SELECT filepath, filename, category, key, bpm, embedding FROM files "
            f"WHERE embedding IS NOT NULL AND filepath != ? AND ({placeholders})",
            [filepath] + [f"%{t}%" for t in target_cats],
        )
        candidates = c2.fetchall()
        conn2.close()

        results = []
        for fp, fn, cat, key, bpm, emb_blob in candidates:
            try:
                emb = _np.frombuffer(emb_blob, dtype=_np.float32)
                sim = float(1.0 - _cosine(query_emb, emb))
                results.append({
                    "filepath": fp,
                    "filename": fn,
                    "category": cat or "",
                    "key": key or "",
                    "bpm": bpm,
                    "similarity": round(sim, 4),
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({"pairs": results[:5], "target_categories": target_cats})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/usage", methods=["GET"])
def usage():
    """METER1 — the honest-state surface: what this user has spent this
    month, against which limits, so the app can SAY when work is paused."""
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not get_user(user_id):
        return jsonify({"error": "user not found"}), 404
    u = get_usage(user_id)
    return jsonify({
        "month": u["month"],
        "embed_count": u["embed_count"],
        "classify_bg_count": u["classify_bg_count"],
        "classify_int_count": u["classify_int_count"],
        "library_size": u["library_size"],
        "search_count": u.get("search_count", 0),
        "describe_count": u.get("describe_count", 0),
        "suggest_count": u.get("suggest_count", 0),
        "intent_count": u.get("intent_count", 0),
        "summarize_count": u.get("summarize_count", 0),
        "first_library_size": (get_user(user_id) or {}).get("first_library_size"),
        "limits": {"embed": EMBED_MONTHLY_LIMIT, "classify": CLASSIFY_MONTHLY_LIMIT,
                   "search": SEARCH_MONTHLY_LIMIT, "suggest": SUGGEST_MONTHLY_LIMIT,
                   "describe": DESCRIBE_MONTHLY_LIMIT, "intent": INTENT_MONTHLY_LIMIT,
                   "summarize": SUMMARIZE_MONTHLY_LIMIT},
        "embed_paused": u["embed_count"] >= EMBED_MONTHLY_LIMIT,
        "classify_paused": u["classify_bg_count"] >= CLASSIFY_MONTHLY_LIMIT,
    })


@app.route("/usage/report", methods=["POST"])
def usage_report():
    """Client reports library size after a scan — the third axis the
    pricing decision needs (spend means nothing without library size)."""
    data = request.json or {}
    user_id = data.get("user_id")
    size = data.get("library_size")
    if not user_id or not isinstance(size, int) or size < 0:
        return jsonify({"error": "user_id and library_size (int) required"}), 400
    if not get_user(user_id):
        return jsonify({"error": "user not found"}), 404
    add_usage(user_id, library_size=size)
    return jsonify({"ok": True})


@app.route("/embed", methods=["POST"])
def embed():
    """Generate text embeddings via Voyage AI for semantic search."""
    data = request.get_json(force=True) or {}
    texts = data.get("texts", [])
    input_type = data.get("input_type", "document")

    if not texts:
        return jsonify({"embeddings": []})

    # METER1 — /embed had NO identity and NO ceiling: an anonymous Voyage
    # proxy whose cost was bounded only by a client-side setting. Identity
    # is now required and the monthly ceiling is checked BEFORE any spend.
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    if not get_user(user_id):
        return jsonify({"error": "user not found"}), 404
    used = get_usage(user_id)["embed_count"]
    if used + len(texts) > EMBED_MONTHLY_LIMIT:
        return jsonify({"error": "monthly_limit", "kind": "embed",
                        "used": used, "limit": EMBED_MONTHLY_LIMIT,
                        "month": month_key()}), 429

    BATCH = 128
    all_embeddings = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i+BATCH]
        try:
            result = VOYAGE_CLIENT.embed(
                chunk,
                model=EMBED_MODEL,
                input_type=input_type,
                output_dimension=EMBED_DIMENSION,
            )
            all_embeddings.extend(result.embeddings)
        except Exception as e:
            print(f"[embed] chunk {i} failed: {e}", flush=True)
            return jsonify({"error": str(e)}), 500

    add_usage(user_id, embed=len(texts))
    return jsonify({
        "embeddings": all_embeddings,
        "model": EMBED_MODEL,
        "dimension": EMBED_DIMENSION,
    })


@app.route("/summarize_project", methods=["POST"])
def summarize_project():
    """Generate a 1-liner project summary for the sidebar.

    Notes are the source of truth (the project brief).
    Chat history layers on evolution — but dominant patterns win
    over one-off tangents.

    Request body:
      {
        "notes": "G minor sad/melancholic with two drops" (str),
        "messages": [
          {"role": "user", "content": "find me an arp"},
          {"role": "assistant", "content": "..."},
          ...
        ]  (list, last ~20 messages, can be empty)
      }

    Response:
      { "summary": "G min · 128 BPM · dark/melancholic" }

    Cost: ~$0.0015 per call (Haiku 4.5).
    """
    data = request.get_json(force=True) or {}
    err, code, _sp_uid = meter_gate(data, "summarize", "summarize_count", SUMMARIZE_MONTHLY_LIMIT)
    if err is not None:
        return err, code
    notes = (data.get("notes") or "").strip()
    messages = data.get("messages", [])

    # If we have neither notes nor messages, return empty — nothing
    # to summarize. Client falls back to first line of notes (which
    # is also empty in this case, so sidebar shows nothing).
    if not notes and not messages:
        return jsonify({"summary": ""})

    # Build the chat transcript section. Cap at last 20 messages
    # to keep token count predictable. Skip empty messages.
    transcript_lines = []
    for m in messages[-20:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        # Trim each message to ~200 chars to stay focused on intent
        if len(content) > 200:
            content = content[:200] + "..."
        prefix = "User" if role == "user" else "AI"
        transcript_lines.append(f"{prefix}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(no chat history yet)"

    notes_block = notes if notes else "(no notes provided)"

    system_prompt = """You are a music producer's assistant generating a glanceable 1-line summary of a song project for a sidebar UI.

The producer needs to scan a list of 5-15 active song projects and immediately remember what each one is about. Your output is the entire 1-liner shown under the project name.

YOU WILL RECEIVE:
1. PROJECT NOTES — the producer's explicit brief (highest priority, treat as source of truth)
2. RECENT CHAT HISTORY — messages between the producer and the AI inside this project (use to surface dominant patterns, NOT to chase recent tangents)

CRITICAL RULES:
- NOTES ARE THE ANCHOR. If notes say "G minor", the summary stays G minor even if recent messages mention other keys.
- DOMINANT PATTERNS WIN. If user discussed key X across 8 messages and key Y in 1 message, summary uses key X. Tangents do not shift the summary.
- BE TERSE. Producer shorthand only. No adjectives like "really" / "kind of" / "very" / "with".
- STRICT FORMAT: <key> · <BPM> · <2-3 vibe tags>
- USE BULLET SEPARATOR: · (middle dot, U+00B7) between sections
- MAX 8 WORDS TOTAL
- ACCEPT ANY KEY FORMAT in input. Producers write keys many ways. ALL of these mean the same key:
    "G", "GMaj", "G Maj", "GMajor", "G Major", "G-Major", "Major G", "g maj", "g major" → all = G major
    "Gm", "Gmin", "G min", "G Minor", "G-Minor", "Minor G", "g minor" → all = G minor
    Same applies to every key (F#, Bb, C#, etc.). When the modifier is missing or ambiguous, default to MAJOR.
- EMIT keys in ONE strict output format only:
    Major keys → "<Note>Maj"  (examples: "GMaj", "F#Maj", "BbMaj", "CMaj", "DMaj")
    Minor keys → "<Note>m"    (examples: "Gm", "F#m", "Bbm", "Cm", "Dm")
- BPM: integer or short range like "128" or "120-130"
- Vibe tags: 1-3 short tags like "dark", "melancholic", "uplifting", "trap", "dubstep", "afro house"

EXAMPLES (note the consistent output format):
"Gm · 128 BPM · dark trap"
"GMaj · 155 BPM · melodic dubstep"
"F#m · 140 BPM · dubstep, melancholic"
"DMaj · 120 BPM · uplifting, melodic"
"Am · 90/180 BPM · afro house"
"Em · ~125 BPM · progressive, uplifting"
"BbMaj · 110 BPM · jazzy, warm"

If you cannot determine a section, omit it. E.g. if no key is clear: "128 BPM · dark trap". If only key is clear: "GMaj · melancholic".

If neither notes nor chat history give you anything to work with, return an empty string.

Respond with ONLY the 1-liner. No quotes, no explanation, no preamble."""

    user_prompt = f"""PROJECT NOTES:
{notes_block}

RECENT CHAT HISTORY (last 20 messages, oldest first):
{transcript}

Generate the 1-liner."""

    try:
        anthropic_client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        response = anthropic_client.messages.create(
            model=SMALL_MODEL,
            max_tokens=60,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as e:
        print(f"[summarize_project] claude call failed: {e}", flush=True)
        return jsonify({"error": f"claude_failed: {e}"}), 500

    # Extract text from the response. Haiku returns text content blocks.
    summary = ""
    for block in response.content:
        if hasattr(block, "text"):
            summary += block.text
    summary = summary.strip()

    # Defensive cleanup: strip surrounding quotes if Claude added them,
    # collapse whitespace, cap length so a misbehaving response can't
    # blow out the sidebar.
    summary = summary.strip('"').strip("'").strip()
    summary = " ".join(summary.split())  # collapse internal whitespace
    if len(summary) > 80:
        summary = summary[:77] + "..."

    print(f"[summarize_project] summary: {summary!r}", flush=True)
    add_usage(_sp_uid, summarize=1)
    return jsonify({"summary": summary})


@app.route("/search", methods=["POST"])
def search():
    """Semantic search: client sends top-50 candidates by cosine similarity,
    Claude re-ranks, writes an explanation, and returns structured filters."""
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    # METER2 — the expensive endpoint (Sonnet), now closed: identity
    # required, monthly ceiling checked before any spend.
    err, code, _search_uid = meter_gate(data, "search", "search_count", SEARCH_MONTHLY_LIMIT)
    if err is not None:
        return err, code
    candidates = data.get("candidates", [])
    conversation = data.get("conversation", [])

    if not query:
        return jsonify({"error": "empty query"}), 400

    if not candidates:
        return jsonify({
            "picks": [],
            "filters_used": {},
            "reply": "Your library has no samples indexed yet.",
            "broad_count": 0,
        })

    candidates_text = "\n".join(
        f"[{c['id']}] {c['meta_text']}"
        for c in candidates[:50]
    )

    # Gate BE1 — system is now TWO blocks, not one concatenated string. The
    # cache breakpoint sits on the stable half; render order is tools ->
    # system, so the marker covers [tools + directives]. The candidate list
    # is volatile per message and must stay AFTER the breakpoint or it would
    # invalidate the very prefix we are trying to cache.
    system_blocks = [
        {
            "type": "text",
            "text": SEARCH_SYSTEM_DIRECTIVES,
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": candidates_text},
    ]

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = conversation + [{"role": "user", "content": query}]

    try:
        response = anthropic_client.messages.create(
            model=SEARCH_MODEL,
            max_tokens=2000,
            # Gate BE1 decision #3 — preserve today's behaviour exactly.
            # Sonnet 5 runs ADAPTIVE thinking when this is omitted (4.6 ran
            # thinking-off), which would share the 2000-token budget with the
            # forced tool call and risk truncating structured output.
            thinking={"type": "disabled"},
            system=system_blocks,
            messages=messages,
            tools=[SEARCH_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "return_search_results"},
        )
    except Exception as e:
        print(f"[search] claude call failed: {e}", flush=True)
        return jsonify({"error": f"claude_failed: {e}"}), 500

    # Extract the tool_use block. With tool_choice forcing this tool, Claude
    # MUST return a tool_use block with validated input matching the schema.
    tool_use = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "return_search_results":
            tool_use = block
            break

    # Gate BE1 decision #5 — report what the call actually cost instead of
    # letting the client assume a flat constant. This is what made the old
    # ai_budget ledger drift (LED1/LED1b): spend was assumed, never measured.
    # It also makes every future model or caching change self-measuring.
    _u = response.usage
    usage_out = {
        "model": SEARCH_MODEL,
        "input_tokens": _u.input_tokens,
        "output_tokens": _u.output_tokens,
        "cache_creation_input_tokens": getattr(_u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(_u, "cache_read_input_tokens", 0) or 0,
    }
    print(f"[search] usage {usage_out}", flush=True)
    add_usage(_search_uid, search=1)

    if tool_use is None:
        print(f"[search] no tool_use block returned. Content: {response.content}", flush=True)
        return jsonify({
            "picks": [],
            "filters_used": {},
            "reply": "Sorry, I had trouble formatting that response. Try rephrasing your search.",
            "broad_count": 0,
            "error": "no_tool_use",
            "usage": usage_out,
        })

    parsed = tool_use.input  # guaranteed dict matching schema

    out = {
        "picks": parsed.get("picks", []),
        "filters_used": parsed.get("filters_used", {}),
        "reply": parsed.get("reply", ""),
        "broad_count": 0,
        "usage": usage_out,
    }
    prog = parsed.get("progression")
    if isinstance(prog, list) and prog:
        out["progression"] = prog
    # FIX4 item 1 — mentions ride the same optional-passthrough shape as
    # progression (C11 precedent): present only when the model produced them,
    # so an older client that ignores the key is unaffected.
    # FIX4b — pass mentions through even when EMPTY. The old `and mentions`
    # made "model returned []" indistinguishable from "model omitted it"
    # client-side, which is exactly the ambiguity that cost FIX4 two sends.
    # The client already treats an empty array as no-chips.
    mentions = parsed.get("mentions")
    if isinstance(mentions, list):
        out["mentions"] = mentions
    print(f"[search] tool keys={sorted(parsed.keys())} mentions={len(mentions) if isinstance(mentions, list) else 'ABSENT'}", flush=True)
    return jsonify(out)




if __name__ == "__main__":
    import sys
    print("Starting Cratify API...", flush=True)
    try:
        init_db()
        print("Database initialized", flush=True)
    except Exception as e:
        print(f"Database error: {e}", flush=True)
        sys.exit(1)
    port = int(os.getenv("PORT", 5000))
    print(f"Running on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
