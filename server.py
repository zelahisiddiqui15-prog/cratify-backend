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
                    deactivate_subscription, set_stripe_customer, hash_password)

load_dotenv()

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

    if not user["subscription_active"]:
        if user["sorts_used"] >= user["trial_limit"]:
            return jsonify({"error": "trial_exhausted"}), 402

    prompt = f"""You are a music file classifier for a producer tool called Cratify.

Analyze this filename and return a JSON object with these fields:
- category: Bass, Lead, Pad, Pluck, FX, Drum, Vocal, Chord, Arp, Guitar, Piano, Strings, Brass, Synth, Texture, Ambient, or Other.
- drum_type: ONLY for Drum: Kick, Snare, Hi-Hat, Clap, Perc, Cymbal, Tom, Full Loop. Null for non-drums.
- subcategory: more specific description
- key: musical key if detectable (e.g. "Am", "C#") or null. Always null for drums.
- bpm: BPM if detectable as number or null
- file_type: "stem", "preset", "midi", "sample", or "loop"
- confidence: 0 to 1

CRITICAL CATEGORY RULES:
Never use Loop as a standalone category. Instead classify by the instrument type:
- Drum Loop, Beat, Break → category: Drum (set drum_type: "Full Loop")
- Bass Loop → category: Bass
- Synth Loop, Synth Riff → category: Synth
- Piano Loop → category: Piano
- Guitar Loop → category: Guitar
- Chord Loop, Chord Stab → category: Chord
- Melody Loop, Lead Loop → category: Lead
- Vocal Loop, Vox Loop → category: Vocal
- Arp Loop → category: Arp
- Pad Loop, Atmosphere Loop → category: Pad
- FX Loop, Riser, Sweep → category: FX
- Texture Loop, Ambient Loop → category: Ambient
- If a loop's instrument is truly unknown: category: Other

Set file_type to "loop" whenever the filename contains "loop", "lp", "riff", "break", or "beat".

Filename: {filename}

Return ONLY valid JSON. No markdown, no explanation."""

    try:
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
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

    # Post-process: drums should never have keys
    if result.get("category") == "Drum":
        result["key"] = None
    # Force preset category for preset file extensions
    ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if ext in ('fxp', 'vital', 'nmsv', 'xpf', 'aupreset', 'patch'):
        result['category'] = 'Preset'
        result['key'] = None
        result['bpm'] = None
        result['file_type'] = 'preset'
    increment_sorts(user_id)
    return jsonify(result)


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
            model="claude-haiku-4-5-20251001",
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

Return a JSON array of 3-4 short prompt strings (each under 12 words,
producer voice, first person, concrete). Rules:
- Each must be an actionable NEXT step for THIS conversation and song.
- Never generic filler; never repeat or trivially rephrase what was just asked.
- If conversation exists, at least 2 suggestions must build on it.
Return ONLY the JSON array, no markdown, no explanation."""

    try:
        message = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
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
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        read = message.content[0].text.strip()
        return jsonify({"read": read})
    except Exception as api_err:
        print(f"[describe_reference] Anthropic API error: {api_err}", flush=True)
        return jsonify({"error": "model_error"}), 502


@app.route("/intent", methods=["POST"])
def intent():
    """Parse a chat message for bulk file action intent using Claude Haiku."""
    data = request.json or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"action": None})
    try:
        resp = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
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


@app.route("/embed", methods=["POST"])
def embed():
    """Generate text embeddings via Voyage AI for semantic search."""
    data = request.get_json(force=True) or {}
    texts = data.get("texts", [])
    input_type = data.get("input_type", "document")

    if not texts:
        return jsonify({"embeddings": []})

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
            model="claude-haiku-4-5-20251001",
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
    return jsonify({"summary": summary})


@app.route("/search", methods=["POST"])
def search():
    """Semantic search: client sends top-50 candidates by cosine similarity,
    Claude re-ranks, writes an explanation, and returns structured filters."""
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
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

    system_prompt = """You are Cratify, an AI music producer assistant. You help producers find the perfect sample from their personal library.

The user will ask for sounds. You have been given the top-50 most semantically similar samples from their library, pre-ranked by vector similarity. Your job is to analyze them, identify the best matches, write a concise producer-friendly explanation, and infer the broader filter criteria for a "see more" button.

Guidelines for your response:
- picks: identify the 4-8 best actual matches. If fewer than 4 truly match, return fewer. If nothing matches well, return empty list.
- reply: write like a text message to a producer friend. Max 3-4 short sentences across 2-3 short paragraphs separated by blank lines. NO run-on sentences. Mention relative major/minor when relevant, pitch-adjust tolerances, layering advice.
- NEVER mention the [ID] numbers in your reply text. Refer to samples by filename or short descriptor ("the F wub one-shot", "that Cm kick").
- filters_used: describes the broader search the user might want ("all vocal chops in Gm around 140 BPM") - be permissive, it's an escape hatch. Any field can be omitted if not inferrable.
- category MUST be one of: Drums, Bass, Synth, Leads, Vocals, FX, Loops, One-Shots, Keys, Percussion, Other
- progression: ONLY when your reply prescribes a chord sequence (e.g. "Fm -> Db -> Ab -> Eb, i -> VI -> III -> VII") AND candidate files match its chords (many filenames carry roman numerals and chord names — e.g. "i - F min.mid", "VI - Db Maj.mid"): populate progression with one entry per step IN PLAYING ORDER, mapping each step to the candidate ids whose filename matches that chord, best first. A step with no matching file gets an empty pick_ids — NEVER force a bad match. Your reply prose is unchanged either way.

You MUST respond by calling the return_search_results tool. Do not respond with text.

The user's library samples (pre-ranked by similarity, ID in brackets):
""" + candidates_text

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
                "filters_used": {
                    "type": "object",
                    "description": "Broader filter criteria the user might want (for 'see more' button).",
                    "properties": {
                        "category": {"type": "string"},
                        "key": {"type": "string"},
                        "bpm_min": {"type": "integer"},
                        "bpm_max": {"type": "integer"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "required": ["picks", "reply", "filters_used"],
        },
    }

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = conversation + [{"role": "user", "content": query}]

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            system=system_prompt,
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

    if tool_use is None:
        print(f"[search] no tool_use block returned. Content: {response.content}", flush=True)
        return jsonify({
            "picks": [],
            "filters_used": {},
            "reply": "Sorry, I had trouble formatting that response. Try rephrasing your search.",
            "broad_count": 0,
            "error": "no_tool_use",
        })

    parsed = tool_use.input  # guaranteed dict matching schema

    out = {
        "picks": parsed.get("picks", []),
        "filters_used": parsed.get("filters_used", {}),
        "reply": parsed.get("reply", ""),
        "broad_count": 0,
    }
    prog = parsed.get("progression")
    if isinstance(prog, list) and prog:
        out["progression"] = prog
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
